import copy
from typing import Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from torch import nn
from transformers import OwlViTConfig
from transformers.models.owlvit.modeling_owlvit import OwlViTVisionTransformer
from transformers.models.detr.modeling_detr import generalized_box_iou, center_to_corners_format
from transformers.image_transforms import corners_to_center_format

from scipy.optimize import linear_sum_assignment
from transformers.utils import requires_backends
from typing import Dict, List, Optional, Tuple
from torch import Tensor, nn


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(self.query_dim, out_dim)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor],
        query_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.FloatTensor]:
        image_class_embeds = self.dense0(image_embeds)
        if query_embeds is None:
            device = image_class_embeds.device
            batch_size, num_patches = image_class_embeds.shape[:2]
            pred_logits = torch.zeros((batch_size, num_patches, self.query_dim)).to(device)
            return (pred_logits, image_class_embeds)

        # Normalize image and text features
        image_class_embeds = F.normalize(image_class_embeds, dim=-1) + 1e-6
        query_embeds = F.normalize(query_embeds, dim=-1) + 1e-6

        # Get class predictions
        pred_logits = torch.einsum("...pd,...qd->...pq", image_class_embeds, query_embeds)

        # Apply a learnable shift and scale to logits
        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        if query_mask is not None:
            if query_mask.ndim > 1:
                query_mask = torch.unsqueeze(query_mask, dim=-2)

            pred_logits = pred_logits.to(torch.float64)
            pred_logits = torch.where(query_mask == 0, -1e6, pred_logits)
            pred_logits = pred_logits.to(torch.float32)

        return (pred_logits, image_class_embeds)


class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, num_layers: int = 3):
        super().__init__()

        print("OwlViTBoxPredictionHead: Number of layers: ", num_layers)

        self.num_layers = num_layers
        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.dense2 = nn.Linear(width, width)
        self.dense3 = nn.Linear(width, width)
        self.dense4 = nn.Linear(width, width)
        self.dense5 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense6 = nn.Linear(width, 4)

    def forward(self, image_features: torch.Tensor) -> torch.FloatTensor:
        output = self.dense0(image_features)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)

        if self.num_layers == 3:
            output = self.dense6(output)
            output = self.gelu(output)
            return output

        output = self.dense2(output)
        output = self.gelu(output)
        output = self.dense3(output)
        output = self.gelu(output)

        if self.num_layers == 5:
            output = self.dense6(output)
            output = self.gelu(output)
            return output

        output = self.dense4(output)
        output = self.gelu(output)
        output = self.dense5(output)
        output = self.gelu(output)
        output = self.dense6(output)
        output = self.gelu(output)

        return output


class OwlViTPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, num_classes: int):
        super().__init__()

        out_dim = config.text_config.hidden_size
        self.query_dim = config.vision_config.hidden_size

        self.num_classes = num_classes

        # Classification projection in paper
        self.mlp_image = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=self.query_dim, out_features=self.query_dim),
            nn.GELU(),
            nn.Linear(in_features=self.query_dim, out_features=self.query_dim),
            nn.GELU(),
            nn.Linear(in_features=self.query_dim, out_features=out_dim),
            nn.GELU(),
        )

    def forward(self,
                image_embeds: torch.FloatTensor,
                query_embeds: torch.FloatTensor,
                topk_idxs: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor]:

        # Get class predictions: topk_idxs (batch_size, n_parts, 1), one_hot (batch_size, n_parts, n_patches*n_patches)
        topk_idxs = torch.swapaxes(topk_idxs, 1, 2)
        one_hot = torch.zeros(topk_idxs.shape[0], topk_idxs.shape[1], image_embeds.shape[1]).to(image_embeds.device).scatter_(2, topk_idxs, 1)
        batch_size, n_parts = one_hot.shape[0], one_hot.shape[1]

        # (batch_size, n_parts, 3600, 1) * (batch_size, 1, 3600, 1024) = (batch_size, n_parts, 3600, 1024).sum(dim=-2)
        image_embeds = (one_hot.unsqueeze(-1) * image_embeds.unsqueeze(1)).sum(dim=-2)

        # image_embeds = self.dense0(image_embeds)            # (batch_size, n_patches, 1024) --> (.., .., 768)
        image_embeds = self.mlp_image(image_embeds.view(-1, image_embeds.shape[-1])).view(batch_size, n_parts, -1)
        query_embeds = query_embeds.view(batch_size, -1, query_embeds.shape[-1])

        # Normalize image and text features
        image_embeds = F.normalize(image_embeds, dim=-1)  # (batch_size, n_parts, 768)
        query_embeds = F.normalize(query_embeds, dim=-1)  # (batch_size, num_classes * n_parts, 768)

        # Send query_embeds to the same device as image_embeds
        query_embeds = query_embeds.to(image_embeds.device)

        # Shape: torch.Size([bs, num_boxes, num_classes * num_parts])
        image_text_logits = torch.einsum('bnd, bid -> bni', image_embeds, query_embeds)
        image_text_logits_reshaped = image_text_logits.view(-1, image_text_logits.shape[-1])

        # Shape: (bs, num_classes * num_parts, num_boxes) --> (bs, num_classes, num_parts, num_boxes)
        pred_logits = image_text_logits.swapaxes(axis0=1, axis1=2).reshape(batch_size, self.num_classes, n_parts, -1) # PEIJIE: Use reshape instead of view to avoid memory issues
        pred_logits = torch.diagonal(pred_logits, dim1=-2, dim2=-1)     # --> torch.Size([bs, num_classes, 12])
        pred_logits = torch.sum(pred_logits, dim=-1)

        return (image_text_logits_reshaped, pred_logits)


class OwlViTForClassification(nn.Module):
    config_class = OwlViTConfig

    def __init__(self, owlvit_det_model, num_classes, num_parts, weight_dict, device, freeze_box_heads=False,
                 train_box_heads_only=False, network_type=None, classification_loss=None, logits_from_teacher=False,
                 finetuning=None, alpha=0.25, gamma=2.0):
        super(OwlViTForClassification, self).__init__()

        self.config = owlvit_det_model.config
        self.num_classes = num_classes
        self.num_parts = num_parts
        self.device = device

        self.alpha = alpha
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction='mean')

        # Use CE loss for classification OR only train with contrastive loss
        self.network_type = network_type
        self.classification_loss = classification_loss
        self.logits_from_teacher = logits_from_teacher
        self.finetuning = finetuning

        # Initialize OwlViT model from the teacher model
        self.owlvit = copy.deepcopy(owlvit_det_model.owlvit)
        self.layer_norm = copy.deepcopy(owlvit_det_model.layer_norm)

        # For image-level classification
        self.cls_head = OwlViTPredictionHead(self.config, self.num_classes)

        # For box prediction
        self.box_head = copy.deepcopy(owlvit_det_model.box_head)

        # For box-level classification
        self.class_head = OwlViTClassPredictionHead(self.config)
        self.class_head.dense0.load_state_dict(owlvit_det_model.class_head.dense0.state_dict())
        self.class_head.logit_shift.load_state_dict(owlvit_det_model.class_head.logit_shift.state_dict())
        self.class_head.logit_scale.load_state_dict(owlvit_det_model.class_head.logit_scale.state_dict())

        # OwlViT: set equal weights for the bounding box, gIoU and classification losses
        # self.matcher = DetrHungarianMatcher(class_cost=1, bbox_cost=1, giou_cost=1)

        # Losses for the criterion in DETR/OwlViT
        self.weight_dict = weight_dict
        losses = ["cardinality"]
        losses += ["boxes"] if weight_dict["loss_bbox"] > 0 else []
        losses += ["labels"] if weight_dict["loss_ce"] > 0 else []

        self.criterion = DetrLoss(
            matcher=None,
            num_parts=self.num_parts,
            eos_coef=0.1,   # Following facebook/detr-resnet-50
            losses=losses,
        )

        self.freeze_parameters(freeze_box_heads, train_box_heads_only, finetuning)
        del owlvit_det_model

    def init_weights_he(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            m.bias.data.fill_(0.01)

    def update_box_head(self, num_layers):
        # TODO: FORCE INITIALIZATION OF BOX HEAD's WEIGHTS
        self.box_head = OwlViTBoxPredictionHead(self.config, num_layers=num_layers)
        self.box_head.apply(self.init_weights_he)

    def freeze_parameters(self, freeze_box_heads, train_box_heads_only, finetuning):
        # OwlViT's text encoder is frozen by default
        for param in self.owlvit.text_model.parameters():
            param.requires_grad = False
        for param in self.owlvit.text_projection.parameters():
            param.requires_grad = False

        # SKIP finetuning box heads
        if freeze_box_heads:
            for param in self.box_head.parameters():
                param.requires_grad = False
            for param in self.class_head.parameters():
                param.requires_grad = False

            # Freeze OwlViT vision encoder
            if finetuning == "mlp_only":
                for param in self.owlvit.parameters():
                    param.requires_grad = False
                for param in self.layer_norm.parameters():
                    param.requires_grad = False

        # SKIP finetuning vision encoder and MLP head for classification --> Adjust weights of box heads only
        if train_box_heads_only:
            for param in self.owlvit.parameters():
                param.requires_grad = False
            for param in self.layer_norm.parameters():
                param.requires_grad = False
            for param in self.cls_head.parameters():
                param.requires_grad = False

    def update_num_classes(self, num_classes):
        self.num_classes = num_classes
        self.cls_head.num_classes = num_classes

    def image_text_embedder(self,
        input_ids: torch.Tensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
    ) -> Tuple[torch.FloatTensor]:

        # Encode text and image
        outputs = self.owlvit(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Get image embeddings
        last_hidden_state = outputs.vision_model_output[0]      # 0: last_hidden_state; 1: pooled_output
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)
        text_embeds = outputs[-4]

        return (text_embeds, image_embeds, outputs)

    def image_embedder(
        self,
        pixel_values: torch.FloatTensor
    ) -> Tuple[torch.FloatTensor]:

        # Get OwlViTModel vision embeddings (same as CLIP)
        vision_outputs = self.owlvit.vision_model(pixel_values=pixel_values, return_dict=True)

        # Apply post_layernorm to last_hidden_state, return non-projected output
        last_hidden_state = vision_outputs[0]
        image_embeds = self.owlvit.vision_model.post_layernorm(last_hidden_state)

        # Resize class token
        new_size = tuple(np.array(image_embeds.shape) - np.array((0, 1, 0)))
        class_token_out = torch.broadcast_to(image_embeds[:, :1, :], new_size)

        # Merge image embedding with class tokens
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        # Resize to [batch_size, num_patches, num_patches, hidden_size]
        new_size = (
            image_embeds.shape[0],
            int(np.sqrt(image_embeds.shape[1])),
            int(np.sqrt(image_embeds.shape[1])),
            image_embeds.shape[-1],
        )
        image_embeds = image_embeds.reshape(new_size)

        return (image_embeds, vision_outputs)

    def normalize_grid_corner_coordinates(self, feature_map: torch.FloatTensor):
        # Computes normalized xy corner coordinates from feature_map.
        if not feature_map.ndim == 4:
            raise ValueError("Expected input shape is [batch_size, num_patches, num_patches, hidden_dim]")

        device = feature_map.device
        num_patches = feature_map.shape[1]

        box_coordinates = np.stack(np.meshgrid(np.arange(1, num_patches + 1), np.arange(1, num_patches + 1)), axis=-1).astype(np.float32)
        box_coordinates /= np.array([num_patches, num_patches], np.float32)

        # Flatten (h, w, 2) -> (h*w, 2)
        box_coordinates = box_coordinates.reshape(box_coordinates.shape[0] * box_coordinates.shape[1], box_coordinates.shape[2])
        box_coordinates = torch.from_numpy(box_coordinates).to(device)

        return box_coordinates

    def compute_box_bias(self, feature_map: torch.FloatTensor) -> torch.FloatTensor:
        # The box center is biased to its position on the feature grid
        box_coordinates = self.normalize_grid_corner_coordinates(feature_map)
        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(-box_coordinates + 1e-4)

        # The box size is biased to the patch size
        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(
        self,
        image_feats: torch.FloatTensor,
        feature_map: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """
        Args:
            image_feats:
                Features extracted from the image, returned by the `image_text_embedder` method.
            feature_map:
                A spatial re-arrangement of image_features, also returned by the `image_text_embedder` method.
        Returns:
            pred_boxes:
                List of predicted boxes (cxcywh normalized to 0, 1) nested within a dictionary.
        """
        # Bounding box detection head [batch_size, num_boxes, 4].
        pred_boxes = self.box_head(image_feats)

        # Compute the location of each token on the grid and use it to compute a bias for the bbox prediction
        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def class_predictor(
        self,
        image_feats: torch.FloatTensor,
        query_embeds: Optional[torch.FloatTensor] = None,
        query_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            image_feats:
                Features extracted from the `image_text_embedder`.
            query_embeds:
                Text query embeddings.
            query_mask:
                Must be provided with query_embeddings. A mask indicating which query embeddings are valid.
        """
        (pred_logits, image_class_embeds) = self.class_head(image_feats, query_embeds, query_mask)

        return (pred_logits, image_class_embeds, query_mask)

    def forward(self, pixel_values, attention_mask, input_ids, text_embeds, targets):

        # Embed images and text queries
        # input_ids = text_inputs_parts["input_ids"]
        # text_embeds_parts, feature_map, outputs = self.image_text_embedder(input_ids=input_ids, attention_mask=text_inputs_parts["attention_mask"],
        #                                                                    pixel_values=image_inputs['pixel_values'])
        text_embeds_parts, feature_map, outputs = self.image_text_embedder(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)

        batch_size, num_patches, num_patches, hidden_dim = feature_map.shape
        image_feats = torch.reshape(feature_map, (batch_size, num_patches * num_patches, hidden_dim))

        # Reshape from [batch_size * max_text_queries, hidden_dim] -> [batch_size, max_text_queries, hidden_dim]
        max_text_queries = input_ids.shape[0] // batch_size
        text_embeds_parts = text_embeds_parts.reshape(batch_size, max_text_queries, text_embeds_parts.shape[-1])

        # If first token is 0, then this is a padded query [batch_size, num_queries].
        input_ids = input_ids.reshape(batch_size, max_text_queries, input_ids.shape[-1])
        query_mask = input_ids[..., 0] > 0

        # Store outputs for computing losses
        loss_dict = {}
        # teacher_boxes_logits = torch.stack([target["logits"] for target in targets], dim=0)     # (bs, num_patches*num_patches, max_text_queries)
        teacher_boxes_logits = targets['logits']     # (bs, num_patches*num_patches, max_text_queries)


        if self.logits_from_teacher:
            pred_logits_parts = None
            topk_scores, topk_idxs = torch.topk(teacher_boxes_logits, k=1, dim=1)
        else:
            # Predict object classes [batch_size, num_patches, num_queries+1]
            pred_logits_parts, class_embeds, _ = self.class_predictor(image_feats, text_embeds_parts, query_mask)

            # Get the top-1 predictions
            scores = self.sigmoid(pred_logits_parts)
            topk_scores, topk_idxs = torch.topk(scores, k=1, dim=1)

        # Predict object boxes (cxcywh) --> Select 12 part boxes out of 576 predicted boxes
        pred_boxes = self.box_predictor(image_feats, feature_map)   # (bs, num_patches*num_patches, 4)
        pred_boxes_selected = torch.gather(pred_boxes, dim=1, index=topk_idxs.squeeze(1).unsqueeze(2).expand(*topk_idxs.squeeze(1).size(), pred_boxes.size(2)))

        # ----------------------------------------------------------------------------------------
        #   Computing box + class (unnecessary) + symmetric losses for box selection
        # ----------------------------------------------------------------------------------------
        outputs_loss = {}
        outputs_loss["logits"] = teacher_boxes_logits if self.logits_from_teacher else pred_logits_parts    # Not necessary as of August 26, 2023
        outputs_loss["pred_boxes"] = pred_boxes

        # Compute box + class losses
        if self.weight_dict["loss_ce"] > 0 or self.weight_dict["loss_bbox"] > 0 or self.weight_dict["loss_giou"] > 0:
            mapping_indices = [(selected_indices, torch.tensor(list(range(self.num_parts))).to(self.device)) for selected_indices in topk_idxs.squeeze(1)]
            loss_dict = self.criterion(outputs_loss, targets, mapping_indices)

        # For getting rid of the teacher model
        if self.weight_dict["loss_sym_box_label"] > 0 and not self.logits_from_teacher:
            # Compute symmetric loss to get rid of the teacher model
            logits_per_image = torch.softmax(pred_logits_parts, dim=1)
            logits_per_text = torch.softmax(pred_logits_parts, dim=-1)

            sym_loss_box_label = self.loss_symmetric(logits_per_image, logits_per_text, teacher_boxes_logits)
            loss_dict["loss_sym_box_label"] = sym_loss_box_label
        # ----------------------------------------------------------------------------------------

        # Predict image-level classes (batch_size, num_patches, num_queries)
        image_text_logits, pred_logits = self.cls_head(image_feats, text_embeds, topk_idxs)
        # targets_cls = torch.tensor([target["targets_cls"] for target in targets]).unsqueeze(1).to(self.device)
        targets_cls = targets["targets_cls"].unsqueeze(1).to(self.device)

        if self.weight_dict["loss_xclip"] > 0:
            if self.network_type == "classification":
                one_hot = torch.zeros_like(pred_logits).scatter(1, targets_cls, 1).to(self.device)

                # Compute CE loss
                if self.classification_loss == "ce_loss":
                    loss = self.ce_loss(pred_logits, one_hot)

                # OR Focal loss
                elif self.classification_loss == "focal_loss":
                    loss = self.focal_loss(pred_logits, targets_cls.squeeze(-1))

                    # Alternative implementation for the focal loss
                    # CE = F.cross_entropy(input=pred_logits, target=one_hot, reduction="none")
                    # prob = torch.gather(torch.softmax(input=pred_logits, dim=-1), 1, targets_cls.long()).squeeze(-1)
                    # loss = (self.alpha * ((1 - prob) ** self.gamma) * CE).mean()

                else:
                    raise f"Loss {self.classification_loss} is not supported"

                loss_dict["loss_xclip"] = loss
            else:

                # Compute symmetric loss for part-descriptor contrastive learning
                logits_per_image = torch.softmax(image_text_logits, dim=0)
                logits_per_text = torch.softmax(image_text_logits, dim=-1)
                sym_loss = self.loss_symmetric(logits_per_image, logits_per_text, targets_cls)
                loss_dict["loss_xclip"] = sym_loss

        return pred_logits, center_to_corners_format(pred_boxes_selected), loss_dict

    def loss_symmetric(self, text_logits: torch.Tensor, image_logits: torch.Tensor, targets: torch.Tensor, box_labels: torch.Tensor = None) -> torch.Tensor:
        # text/image logits (batch_size*num_boxes, num_classes*num_descs): The logits that softmax over text descriptors or boxes
        # targets (batch_size, 1): The ground truth label of box-text pair for classification OR
        # targets (batch_size, all_boxes, num_parts): The ground truth label of box-text pair for box selection
        # box_labels (batch_size, num_boxes), 0 for no box, 1 for box

        # to support DP, define local device
        device = text_logits.device
        assert text_logits.shape == image_logits.shape # should not assert this.

        # For image classification
        if image_logits.shape != targets.shape:
            batch_size = targets.shape[0]

            # get the matching labels (bs * 12, num_classes * num_parts)
            default_box_labels = torch.kron(torch.ones(batch_size, self.num_classes), torch.eye(self.num_parts)).to(device)
            if box_labels is None:
                box_labels = default_box_labels.clone()
            else:
                # (batch_size, num_boxes) -> (bs * num_boxes, num_classes * num_parts)
                box_labels = box_labels.view(-1, 1) * default_box_labels

            # Create one-hot encoding of targets; matching_labels shape: (bs * 12, num_classes * num_parts)
            target_one_hot = torch.zeros(batch_size, self.num_classes).to(device).scatter(1, targets.reshape(-1, 1), 1)
            target_one_hot = torch.kron(target_one_hot, torch.ones(self.num_parts, self.num_parts).to(device))

            matching_labels = target_one_hot * box_labels
        else:
            # For box selection: matching_labels shape: (bs, 576, num_parts)
            values, indices = torch.max(targets, dim=1)
            matching_labels = torch.zeros_like(targets).scatter(1, indices.unsqueeze(1), 1)

        loss_i = F.cross_entropy(image_logits, matching_labels.to(device), reduction='mean')
        loss_t = F.cross_entropy(text_logits, matching_labels.to(device), reduction='mean')
        sym_loss = (loss_i + loss_t).mean()

        return sym_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        # Compute the Softmax and Cross-Entropy Loss
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Convert the target labels to one-hot format
        num_classes = input.size(1)  # Number of classes
        target_onehot = torch.zeros_like(input)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        # Compute the probabilities of the true labels
        prob = F.softmax(input, dim=1)
        true_prob = torch.sum(prob * target_onehot, dim=1)

        # Compute the Focal Loss
        focal_weight = (self.alpha * ((1 - true_prob) ** self.gamma))
        focal_loss = focal_weight * ce_loss

        # Apply the reduction method (mean, sum, or none)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class DetrHungarianMatcher(nn.Module):
    """
    This class computes an assignment between the targets and the predictions of the network.

    For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    un-matched (and thus treated as non-objects).

    Args:
        class_cost:
            The relative weight of the classification error in the matching cost.
        bbox_cost:
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost:
            The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])

        self.class_cost = class_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        target_ids = torch.cat([v["class_labels"] for v in targets])
        target_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        class_cost = -out_prob[:, target_ids]

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

        # Compute the giou cost between boxes
        giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


class DetrLoss(nn.Module):
    """
    This class computes the losses for DetrForObjectDetection/DetrForSegmentation. The process happens in two steps: 1)
    we compute hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair
    of matched ground-truth / prediction (supervise class and box).

    A note on the `num_classes` argument (copied from original repo in detr.py): "the naming of the `num_classes`
    parameter of the criterion is somewhat misleading. It indeed corresponds to `max_obj_id` + 1, where `max_obj_id` is
    the maximum id for a class in your dataset. For example, COCO has a `max_obj_id` of 90, so we pass `num_classes` to
    be 91. As another example, for a dataset that has a single class with `id` 1, you should pass `num_classes` to be 2
    (`max_obj_id` + 1). For more details on this, check the following discussion
    https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223"


    Args:
        matcher (`DetrHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_parts (`int`):
            Number of object categories, omitting the special no-object category.
        eos_coef (`float`):
            Relative classification weight applied to the no-object category.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
    """

    def __init__(self, matcher, num_parts, eos_coef, losses):
        super().__init__()
        self.matcher = matcher
        self.num_parts = num_parts
        self.eos_coef = eos_coef
        self.losses = losses

        # empty_weight = torch.ones(self.num_parts + 1)
        empty_weight = torch.ones(self.num_parts)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

    # NOTE: WE DO NOT USE BOX LABEL LOSS IN OUR METHOD
    # Removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (NLL) targets dicts must contain the key "class_labels" containing a tensor of dim
        [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        # target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)])
        # target_classes = torch.full(source_logits.shape[:2], self.num_parts, dtype=torch.int64, device=source_logits.device)
        # target_classes[idx] = target_classes_o

        source_logits = source_logits[idx].view(len(indices), -1, self.num_parts)
        # target_classes = torch.stack([t["class_labels"][J] for t, (_, J) in zip(targets, indices)], dim=0)
        
        # This will create a 1D tensor with the within_batch_indices from indices
        within_batch_indices = torch.tensor([J for _, J in indices], device=source_logits.device)
        # Now, index into the batched 'labels' tensor using these within_batch_indices
        target_classes = targets['labels'][:, within_batch_indices]

        loss_ce = F.cross_entropy(source_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(targets["class_labels"])], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]. The target boxes
        are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")

        device = outputs["pred_boxes"].device

        idx = self._get_source_permutation_idx(indices)
        # source_boxes = center_to_corners_format(outputs["pred_boxes"][idx])                         # corner format
        # target_boxes = torch.cat([torch.tensor(t["boxes"]).to(device) for t in targets], dim=0)     # corner format

        source_boxes = outputs["pred_boxes"][idx]                                                     # center format
        # target_boxes = torch.cat([torch.tensor(t["boxes"]).to(device) for t in targets], dim=0)       # corner format
        target_boxes = targets['boxes'].reshape(-1, 4)                                                # corner format
        target_boxes = corners_to_center_format(target_boxes)                                         # center format

        losses = {}

        loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="mean")
        losses["loss_bbox"] = loss_bbox

        # Since we have part-part mappings already, the giou loss for each part is from the diagonal of the gIoU matrix
        loss_giou = 1 - torch.diag(generalized_box_iou(center_to_corners_format(source_boxes), center_to_corners_format(target_boxes)))
        losses["loss_giou"] = loss_giou.mean()

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.

        Targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w].
        """
        if "pred_masks" not in outputs:
            raise KeyError("No predicted masks found in outputs")

        source_idx = self._get_source_permutation_idx(indices)
        target_idx = self._get_target_permutation_idx(indices)
        source_masks = outputs["pred_masks"]
        source_masks = source_masks[source_idx]
        masks = [t["masks"] for t in targets]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(source_masks)
        target_masks = target_masks[target_idx]

        # upsample predictions to the target size
        source_masks = nn.functional.interpolate(
            source_masks[:, None], size=target_masks.shape[-2:], mode="bilinear", align_corners=False
        )
        source_masks = source_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(source_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(source_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(source_masks, target_masks, num_boxes),
        }
        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets, indices):
        """
        This performs the loss computation.

        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}

        # ThangPM: Do NOT use bipartite matching --> Use the boxes selected by argmax for computing symmetric loss
        # Retrieve the matching between the outputs of the last layer and the targets
        # indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across all nodes, for normalization purposes
        # num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = len(targets["class_labels"])
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=outputs["pred_boxes"].device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                # indices = self.matcher(auxiliary_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    def _max_by_axis(the_list):
        # type: (List[List[int]]) -> List[int]
        maxes = the_list[0]
        for sublist in the_list[1:]:
            for index, item in enumerate(sublist):
                maxes[index] = max(maxes[index], item)
        return maxes

    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

from torch.distributed import all_gather


