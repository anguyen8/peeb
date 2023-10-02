from typing import Dict, List, Optional, Tuple
from torch import Tensor
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
from torchvision.transforms.autoaugment import _apply_op
import torch


class TrivialAugmentation(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`_.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
    """
    available_ops = ['Identity', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate', 'Brightness', 'Color', 'Contrast', 'Sharpness', 'Posterize', 'Solarize', 'AutoContrast', 'Equalize']

    def __init__(
        self,
        num_magnitude_bins: int = 31,
        interpolation: InterpolationMode = InterpolationMode.NEAREST,
        fill: Optional[List[float]] = None,
        operations: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill
        self.op_meta = self._augmentation_space(num_bins=num_magnitude_bins, operations=operations)
        # if operations is not None, check if the operations are valid
        if operations is not None:
            for op in operations:
                assert op in self.available_ops, f"Operation {op} is not supported. Please choose from {self.available_ops}"

    def _augmentation_space(self, num_bins: int, operations: List[str]) -> Dict[str, Tuple[Tensor, bool]]:
        all_op = {
            # op_name: (magnitudes, signed)
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
        if operations is None:
            return all_op
        else:
            return {k: v for k, v in all_op.items() if k in operations}

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.

        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        channels, height, width = F.get_dimensions(img)
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * channels
            elif fill is not None:
                fill = [float(f) for f in fill]

        # op_meta = self._augmentation_space(self.num_magnitude_bins) # moved to __init__
        op_index = int(torch.randint(len(self.op_meta), (1,)).item())
        op_name = list(self.op_meta.keys())[op_index]
        magnitudes, signed = self.op_meta[op_name]

        # PEIJIE: fix magnitudes for float images input in Solarize opertation
        magnitudes = magnitudes.float() / 255.0 if op_name == 'Solarize' else magnitudes
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_magnitude_bins={self.num_magnitude_bins}"
            f", interpolation={self.interpolation}"
            f", fill={self.fill}"
            f")"
        )
        return s


