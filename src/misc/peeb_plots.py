import io
import os
import copy
import glob
import json
import base64
import pdfkit
import textwrap
import argparse
from pathlib import Path

import torch
import fire
import pdf2image
import numpy as np
from tqdm import tqdm
from matplotlib import gridspec, patheffects
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from .plot_tools import get_pre_define_colors, Drawer
from .iou_tools import normalize_boxes, compute_giou, compute_l1_loss


def rgb_to_hex(rgb):
    return f"#{''.join(f'{x:02x}' for x in rgb)}"

ORDERED_PARTS = ['crown', 'forehead', 'nape', 'eyes', 'beak', 'throat', 'breast', 'belly', 'back', 'wings', 'legs', 'tail']
ORG_PART_ORDER = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']
COLORS = get_pre_define_colors(12, is_float=True, cmap_set=['Set2', 'tab10'])
PART_COLORS = {part: COLORS[i] for i, part in enumerate(ORDERED_PARTS)}
COLORS_INT = get_pre_define_colors(12, is_float=False, cmap_set=['Set2', 'tab10'])
PART_COLORS_INT = {part: COLORS_INT[i] for i, part in enumerate(ORDERED_PARTS)}
# use light blue for all descriptors for sachit (in float)
SACHIT_COLOR = (0.2549019607843137, 0.4117647058823529, 0.8823529411764706)
SACHIT_COLOR_HTML = "#ADD8E6"
XCLIP_GTP4_DESC = json.load(open('data/class_lists/descriptors_bird_soup_v21.json', 'r'))
DESCRIPTION_PART_ORDER = ['back', 'beak', 'belly', 'breast', 'crown', 'forehead', 'eyes', 'legs', 'wings', 'nape', 'tail', 'throat']
HTML_COLORS = {part: rgb_to_hex(COLORS_INT[i]) for i, part in enumerate(ORDERED_PARTS)}

def fig_to_pil(fig, out_format: str = 'png'):
    """Convert a Matplotlib figure to a PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="out_format", bbox_inches='tight', pad_inches=0, dpi=300)  # Use high dpi for better quality
    buf.seek(0)
    img = Image.open(buf)
    return img


def get_exp(descriptions: list[str], 
            scores: list[float], 
            figure_size: tuple[int, int] = (12, 7), 
            fontsize: int = 16, 
            is_visible: list[bool]=None, 
            colors: list[float] = None, 
            out_format: str = 'jpg',
            bar_width: float = 0.3):
    # Filter by visibility, if provided
    if is_visible is not None:
        colors = [color for i, color in enumerate(colors) if is_visible[i]]
        descriptions = [desc for i, desc in enumerate(descriptions) if is_visible[i]]
        scores = [score for i, score in enumerate(scores) if is_visible[i]]

    # make sure text are in TrueType font (to save as pdf)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Create a new figure
    fig, ax = plt.subplots(figsize=figure_size)  # Adjust as needed

    # Generate a horizontal bar plot with reduced bar width
    y_pos = np.arange(len(descriptions))
    bars = ax.barh(y_pos, scores, align='center', color=colors, height=bar_width)  # Use the generated colors

    # Invert the y-axis so the item with the highest score is at the top
    ax.invert_yaxis()

    # Remove the axes
    ax.axis('off')

    # Add the scores on the bars
    for i, (v, bar, color) in enumerate(zip(scores, bars, colors)):
        ax.text(bar.get_width() + 0.02, i + 0.05, f"{v:.3f}", color=color, va='center', ha='left', fontweight='bold', fontsize=fontsize)

    # Add the descriptions above the bars
    for i, desc in enumerate(descriptions):
        ax.text(0, i - 0.2, desc, color='black', va='bottom', ha='left', fontsize=fontsize, fontweight='bold')

    # Ensure layout fits
    plt.tight_layout()
    
    match out_format:
        case 'png' | 'jpg':
            out_image = fig_to_pil(fig)
            plt.close(fig)
            return out_image
        
        case 'pdf':
            pdf_stream = io.BytesIO()
            fig.savefig(pdf_stream, format="pdf")
            pdf_content = pdf_stream.getvalue()
            plt.close(fig)
            pdf_stream.close()
            return pdf_content
        
        # note: fig is not closed here, make sure close it after use
        case __:
            return fig 


def get_exp_overlay(descriptions: list[str], 
                    scores: list[float], 
                    figure_size: tuple[int, int] = (8, 6), 
                    fontsize: int = 14, 
                    is_visible: list[bool]=None, 
                    colors: list[float] = None, 
                    out_format: str = 'jpg',
                    bar_width: float = 0.8,
                    max_text_length: int = 60):
    # Filter by visibility, if provided
    if is_visible is not None:
        colors = [color for i, color in enumerate(colors) if is_visible[i]]
        descriptions = [desc for i, desc in enumerate(descriptions) if is_visible[i]]
        scores = [score for i, score in enumerate(scores) if is_visible[i]]

    # make sure text are in TrueType font (to save as pdf)
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['ps.fonttype'] = 42

    # Create a new figure
    fig, ax = plt.subplots(figsize=figure_size)  # Adjust as needed

    # Generate a horizontal bar plot with reduced bar width
    y_pos = np.arange(len(descriptions))
    bars = ax.barh(y_pos, scores, align='center', color=colors, height=bar_width)  # Use the generated colors
    ax.set_xlim(0, 1)  # Explicitly set x-axis limits to [0, 1]
    # Invert the y-axis so the item with the highest score is at the top
    ax.invert_yaxis()

    # Remove the axes
    ax.axis('off')

    rightmost_limit = ax.get_xlim()[1]  # Get the rightmost edge of the plot
    
    # Scale all bars by the scaling factor
    for bar in bars:
        bar.set_width(bar.get_width() * 0.9)
    

    rightmost_limit = ax.get_xlim()[1]  # Get the rightmost edge of the plot
    # Add the scores on the bars
    for i, (v, bar, color) in enumerate(zip(scores, bars, colors)):
        ax.text(rightmost_limit - 0.02, i + 0.05, f"{v:.3f}", color=color, va='center', ha='right', fontweight='bold', fontsize=fontsize)

    # Adjust font size based on the length of the longest description
    max_desc_length = max(map(len, descriptions))
    if max_desc_length > max_text_length:  # Arbitrary threshold, adjust as needed
        fontsize -= (max_desc_length - max_text_length) * 0.1  # Scale font size based on excess length

    # Ensure font size doesn't get too small
    fontsize = max(fontsize, 8)  # Set a floor for the smallest font size

    # Add the descriptions above the bars
    for i, desc in enumerate(descriptions):
        ax.text(0, i, desc, color='black', va='center', ha='left', fontsize=fontsize, fontweight='bold')

    # Ensure layout fits
    plt.tight_layout()
    
    match out_format:
        case 'png' | 'jpg':
            """Convert a Matplotlib figure to a PIL Image."""
            buf = io.BytesIO()
            fig.savefig(buf, format=out_format, bbox_inches='tight', pad_inches=0, dpi=300)  # Use high dpi for better quality
            buf.seek(0)
            img = Image.open(buf)
            return img
        
        case 'pdf':
            pdf_stream = io.BytesIO()
            fig.savefig(pdf_stream, format="pdf")
            pdf_content = pdf_stream.getvalue()
            plt.close(fig)
            pdf_stream.close()
            return pdf_content
        
        # note: fig is not closed here, make sure close it after use
        case __:
            return fig, ax

def results_confusion_matrix(result_dict1: dict, result_dict2: str, methods: list[str] = ["sachit", "xclip"], file_name_only: bool = False)-> dict:
    
    if isinstance(result_dict1, str):
        method1 = json.load(open(result_dict1, 'r'))
        method2 = json.load(open(result_dict2, 'r'))
        # use file_name as dictionary key
        method1 = {item['file_name']: item for item in method1}
        method2 = {item['file_name']: item for item in method2}
    else:
        method1 = result_dict1
        method2 = result_dict2
    # get intersection of two dictionaries
    intersect_files = set(method1.keys()).intersection(set(method2.keys()))
    
    # get the "confusion matrix", i.e. both correct, both wrong, one correct one wrong
    # use two letter naming convention, first letter is method1, second letter is method2. e.g., TF means method1 is correct, method2 is wrong
    confusion_dict = {'TT': [], 'TF': [], 'FT': [], 'FF': []}
    for file_name in intersect_files:
        result1 = method1[file_name]
        result2 = method2[file_name]
        group_dict = file_name if file_name_only else {f'{methods[0]}': result1, f'{methods[1]}': result2}
        if result1['prediction'] and result2['prediction']:
            confusion_dict['TT'].append(group_dict)
        elif result1['prediction'] and not result2['prediction']:
            confusion_dict['TF'].append(group_dict)
        elif not result1['prediction'] and result2['prediction']:
            confusion_dict['FT'].append(group_dict)
        else:
            confusion_dict['FF'].append(group_dict)
    return confusion_dict



def colored_title(prediction: str, ground_truth: str, scores: float) -> tuple:
    """
    Returns the prediction text and the color for the prediction.
    The prediction text will be colored based on its match with the ground truth.
    """
    color = 'green' if prediction == ground_truth else 'red'
    pred_text = f"{prediction} | {scores:.3f}"
    return pred_text, color


def set_colored_title(ax, prediction, ground_truth, scores, base_text: str = None, fontsize: int = None):
    # Determine the color based on prediction
    color = 'green' if prediction == ground_truth else 'red'

    if base_text:
        # Set the title with the base_text
        title = ax.set_title(base_text, color="black", y=1.05)  # Slight y adjustment for spacing

        # Get title's font properties
        title_fontsize = fontsize or title.get_fontsize()

    else:
        # No base text, so we use the top of the axis for prediction
        title_fontsize = fontsize or ax.title.get_fontsize()  # Get the default title fontsize
        # Clear any existing title
        ax.set_title("")

    # Position for prediction text below the title
    ypos = 1.02
    # Use ax.text to add the prediction text at the specified position in axis coordinates
    ax.text(0.5, ypos,  # Position (0.5, 1) corresponds to the center-top of the axis
            f"{prediction} | {scores:.3f}", 
            color=color, 
            ha='center',
            va='baseline',
            fontsize=title_fontsize,
            transform=ax.transAxes)  # Using axis coordinates

    return ax


def pdf_content_to_array(pdf_content):
    images = pdf2image.convert_from_bytes(pdf_content)
    if images:
        return np.asarray(images[0])
    else:
        return None

def add_title_to_image(img, title):
    font_size = 20
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    
    # Calculate the size required for the text and create a new image with extra space
    _, _, text_width, text_height = font.getbbox(title)
    new_img = Image.new('RGB', (img.width, img.height + text_height + 10), 'white')
    
    # Paste the original image below the space for the title
    new_img.paste(img, (0, text_height + 10))
    
    # Draw the title
    draw = ImageDraw.Draw(new_img)
    text_x = (img.width - text_width) // 2  # center the text
    draw.text((text_x, 5), title, font=font, fill="black")
    
    return new_img

def concat_images(img1, img2, exp1, exp2, title1, title2, exp_title1, exp_title2):
    # concat image into 2x2 grid:
    # img1 | exp1
    # img2 | exp2

    # Find the maximum height between exp1 and exp2
    max_height = max(exp1.height, exp2.height)

    # Resize exp1 and exp2 to have the same height
    exp1, exp2 = resize_images_to_match_height(exp1, exp2, max_height)

    # Now, scale both exp1 and exp2 to match the height of img1
    exp1, exp2 = resize_images_to_match_height(exp1, exp2, img1.height)
    
    # Add the titles to the images
    img1 = add_title_to_image(img1, title1)
    img2 = add_title_to_image(img2, title2)
    exp1 = add_title_to_image(exp1, exp_title1)
    exp2 = add_title_to_image(exp2, exp_title2)
    
    # Calculate the size of the output image
    width = img1.width + exp1.width
    height = img1.height + img2.height
    
    # Create a new image with the calculated size
    new_img = Image.new('RGB', (width, height), 'white')
    
    # Paste each image into the correct position
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (0, img1.height))
    new_img.paste(exp1, (img1.width, 0))
    new_img.paste(exp2, (img2.width, img1.height))
    
    return new_img

def resize_image(img, fixed_width):
    """Resize an image to a fixed width while maintaining its aspect ratio."""
    aspect_ratio = fixed_width / float(img.width)
    new_height = int(aspect_ratio * float(img.height))
    return img.resize((fixed_width, new_height), Image.Resampling.LANCZOS)

def resize_images_to_match_height(img1, img2, target_height):
    """Resize images to match a target height while maintaining their aspect ratios."""
    aspect_ratio1 = target_height / float(img1.height)
    new_width1 = int(aspect_ratio1 * float(img1.width))
    img1_resized = img1.resize((new_width1, target_height), Image.Resampling.LANCZOS)

    aspect_ratio2 = target_height / float(img2.height)
    new_width2 = int(aspect_ratio2 * float(img2.width))
    img2_resized = img2.resize((new_width2, target_height), Image.Resampling.LANCZOS)

    return img1_resized, img2_resized

def generate_analitical_pdf(num_samples: int, 
                            org_image_folder: str, 
                            box_image_folder: str, 
                            out_dir: str, 
                            result_dicts: dict, 
                            result_type: str = None, 
                            hide_invisible: bool = True, 
                            out_format: str = 'jpg',
                            use_teacher_logits: bool = True,
                            methods: list[str] = ["sachit", "xclip"],
                            show_gt_desc: bool = False, # show descriptions for ground truth disregard of the correctness of the prediction. (only apply to xclip)
                            image_1: str = "org_image",
                            use_birdsoup_image: bool = False,
                            random_seed: int = 42,
                            show_gt: bool = True,
                            ):
    visibility_dict = json.load(open('data/cub_vis_binary.json', 'r'))
    os.makedirs(out_dir, exist_ok=True)
    plot_dict = {}
    total_samples = 0
    np.random.seed(random_seed)
    if result_type is None:
        # randomly sample num_samples items per category (if possible)
        for category in result_dicts:
            if len(result_dicts[category]) == 0:
                continue
            elif len(result_dicts[category]) < num_samples:
                plot_dict[category] = result_dicts[category]
            else: 
                plot_dict[category] = np.random.choice(result_dicts[category], num_samples, replace=False)
            total_samples += len(plot_dict[category])
    else:
        for category in result_type:
            if len(result_dicts[category]) == 0:
                continue
            elif len(result_dicts[category]) < num_samples:
                plot_dict[category] = result_dicts[category]
            else: 
                plot_dict[category] = np.random.choice(result_dicts[category], num_samples, replace=False)
            total_samples += len(plot_dict[category])


    # generate explanations for each sample (for both model prediction and ground truth)
    pbar = tqdm(total=total_samples, desc="Generating explanations")
    for category, plot_list in plot_dict.items():
        
        for idx, result_item in enumerate(plot_list):
            image_name = result_item[methods[0]]['bs_file_name'] if use_birdsoup_image else result_item[methods[0]]['file_name']
            base_name = Path(image_name).stem

            visibility = visibility_dict[image_name] if hide_invisible else {part: True for part in ORDERED_PARTS}
            gt_label = result_item[methods[1]]['ground_truth']

            method_results = {}
            colors = {}
            visibility_list = {}
            for method_name in methods:
                if method_name == "sachit":
                    pred = result_item[method_name]['pred']
                    desc = result_item[method_name]['desc']
                    scores = result_item[method_name]['scores']
                    # sort the scores and descriptions by the scores
                    desc = [desc for _, desc in sorted(zip(scores, desc), reverse=True)]
                    scores = sorted(scores, reverse=True)
                    method_results[method_name] = {'pred': pred, 'desc': desc, 'scores': scores}
                    colors[method_name] = [SACHIT_COLOR] * len(desc)
                else:
                    if show_gt_desc and method_name == 'xclip':
                        desc = XCLIP_GTP4_DESC[gt_label]
                        desc = dict(zip(DESCRIPTION_PART_ORDER, desc))
                        scores = result_item[method_name]['gt_scores']
                        pred = result_item[method_name]['ground_truth']
                    else:
                        desc = result_item[method_name]['descriptions']
                        scores = result_item[method_name]['pred_scores']
                        pred = result_item[method_name]['pred']
                    desc = [desc[part] for part in ORDERED_PARTS]
                    scores = [scores[part] if scores[part] > 0 else 0 for part in ORDERED_PARTS]
                    colors[method_name] = list(PART_COLORS.values())
                    visibility_list[method_name] = [visibility[part] for part in ORDERED_PARTS]
                    method_results[method_name] = {'pred': pred, 'desc': desc, 'scores': scores}

            if methods[0] == "sachit":
                exp1 = get_exp(descriptions=method_results[methods[0]]['desc'], scores=method_results[methods[0]]['scores'], colors=colors[methods[0]])
            else:
                exp1 = get_exp(descriptions=method_results[methods[0]]['desc'], scores=method_results[methods[0]]['scores'], colors=colors[methods[0]], is_visible=visibility_list[methods[0]])
            exp2 = get_exp(descriptions=method_results[methods[1]]['desc'], scores=method_results[methods[1]]['scores'], colors=colors[methods[1]], is_visible=visibility_list[methods[1]])


            original_image = Image.open(os.path.join(org_image_folder, image_name))
            if hide_invisible:
                box_file_name = f"{base_name}_visible.jpg"
            else:
                box_file_name = f"{base_name}_all.jpg"
            
            box_image_path = os.path.join(box_image_folder, box_file_name)
            if os.path.exists(box_image_path):
                teacher_boxes_image = Image.open(box_image_path)

            if not use_teacher_logits:
                image_with_boxes = {}
                for i in range(len(methods)):
                    if methods[i] == 'sachit':
                        image_with_boxes[i] = original_image
                    else:
                        boxes = result_item[methods[i]]['pred_boxes']
                        boxes = {part: boxes[part] for part in ORDERED_PARTS}
                        boxes = [box for part, box in boxes.items() if visibility[part]]
                        part_colors = [color for part_name, color in PART_COLORS_INT.items() if visibility[part_name]]
                        image_with_boxes[i] = Drawer.draw_boxes(original_image, boxes, part_colors, width=2)
   
            output_pdf_path = os.path.join(out_dir, f"{category}_{base_name}.{out_format}")

            if image_1 == "org_image":
                image1 = original_image
            elif image_1 == "teacher_boxes":
                image1 = teacher_boxes_image
            else:
                image1 = image_with_boxes[0]
            
            out_image = concat_images(image1, image_with_boxes[1], exp1, exp2, f"GT: {gt_label}", " ", f"{methods[0]}: {method_results[methods[0]]['pred']}", f"{methods[1]}: {method_results[methods[1]]['pred']}")
            out_image.save(output_pdf_path, quality=100, subsampling=0)

            pbar.update(1)


def generate_comparisons(result_dict1: str, result_dict2: str):
    sachit_results = "runs/logits/sachit_cub/sachit_logits.json"
    xclip_results = "runs/logits/xclip_birdsoupv2/cub_test/level_1_e018_Aug_21/xclip_pred.json"
    xclip_results_revised = "runs/logits/xclip_birdsoupv2/cub_test/step2_level1_e5_cub_183_revised_desc/xclip_pred.json"

    org_image_folder = 'runs/explanations/cub_test/org'
    # org_image_folder = '/home/lab/datasets/bird_soup/images'
    box_image_folder = 'runs/explanations/cub_test/boxes'
    
    methods = ["xclip", "xclip_revised2"]
    out_folder_subfix = "pretrain_gt2"
    
    result_type = ["TF", "TT"]
    use_teacher_logits = False
    show_org_image = False
    show_gt_desc = False # for ablation study only, show ground truth descriptions of "xclip" regardless of the correctness of the prediction. 
    hide_invisible = False
    use_birdsoup_image = False
    image_1 = "org_image"
    
    out_dir = f'runs/paper_plots/cub_test/{methods[0]}_vs_{methods[1]}'
    # if out_folder_subfix is not None or not empty string, add it to the output folder name
    if out_folder_subfix is not None and out_folder_subfix != "":
        out_dir = f'{out_dir}_{out_folder_subfix}'
        
    if methods == ["sachit", "xclip"]:
        result_dict1 = sachit_results
        result_dict2 = xclip_results
    elif methods == ["xclip", "xclip_revised"]:
        result_dict1 = xclip_results
        result_dict2 = xclip_results_revised
    elif methods == ["xclip", "xclip_revised2"]:
        result_dict1 = "runs/logits/xclip_birdsoupv2/cub_test/editability_level1/xclip_pred.json"
        result_dict2 = "runs/logits/xclip_birdsoupv2/cub_test/editability_level1_revised/xclip_pred.json"

    
    confusion_dict = results_confusion_matrix(result_dict1, result_dict2, methods=methods)
    generate_analitical_pdf(num_samples=10, 
                             org_image_folder=org_image_folder,
                             box_image_folder=box_image_folder,
                             out_dir=out_dir,
                             result_dicts=confusion_dict,
                             result_type=result_type,
                             hide_invisible=hide_invisible,
                             out_format='jpg',
                             methods=methods,
                             use_teacher_logits=use_teacher_logits,
                             show_gt_desc=show_gt_desc,
                             image_1=image_1,
                             use_birdsoup_image=use_birdsoup_image,
                             )

def generate_explainations(result_dict: str = "runs/logits/xclip_birdsoupv2/cub_183/revised/xclip_pred.json", 
                           out_format: str = 'pdf', 
                           out_folder: str = 'temp/xclip_cub_exps_183', 
                           num_samples: int = None,
                           method: str = 'xclip',
                           visible_only: bool = False,
                           class_name: str = None,):
    os.makedirs(f'{out_folder}/correct', exist_ok=True)
    os.makedirs(f'{out_folder}/wrong', exist_ok=True)

    # load the results
    result_dict = json.load(open(result_dict, 'r'))
    result_dict = [item for item in result_dict if item['prediction']] # only keep correct predictions
    if class_name is not None:
        result_dict = [item for item in result_dict if item['ground_truth'] == class_name]

    visibility_dict = json.load(open('data/cub_vis_binary.json', 'r'))

    for idx, item in tqdm(enumerate(result_dict), desc="Generating explanations", total=len(result_dict) if num_samples is None else num_samples):
        if num_samples is not None and idx >= num_samples:
            break
        
        file_name = item['file_name']
        base_name = Path(file_name).stem
        visibility = visibility_dict[item['file_name']] if visible_only else {part: True for part in ORDERED_PARTS}
        visibility_list = [visibility[part] for part in ORDERED_PARTS]
        if method == "sachit":
            desc = item['desc']
            colors = [SACHIT_COLOR] * len(desc)
            exp = get_exp_overlay(descriptions=desc, scores=item['pred_scores'], colors=colors)
        else:
            colors = list(PART_COLORS.values())
            scores_list = [max(item['pred_scores'][part], 0) for part in ORDERED_PARTS]
            desc_list = [item['descriptions'][part] for part in ORDERED_PARTS]
            exp = get_exp_overlay(descriptions=desc_list, scores=scores_list, colors=colors, is_visible=visibility_list, out_format='pdf')

        if item['prediction']:
            with open(os.path.join(out_folder, 'correct',f"{base_name}.{out_format}"), 'wb') as f:
                f.write(exp)
        else:
            with open(os.path.join(out_folder, 'wrong', f"{base_name}.{out_format}"), 'wb') as f:
                f.write(exp)

def get_exp_overlay2(ax, descriptions: list[str], scores: list[float], fontsize: int = 14,
                     is_visible: list[bool] = None, colors: list[float] = None, bar_width: float = 0.8,
                     max_text_length: int = 60, bar_max: ['auto', 'absolute'] = 'auto'):
    # Filter by visibility, if provided
    if is_visible is not None:
        colors = [color for i, color in enumerate(colors) if is_visible[i]]
        descriptions = [desc for i, desc in enumerate(descriptions) if is_visible[i]]
        scores = [score for i, score in enumerate(scores) if is_visible[i]]

    # Generate a horizontal bar plot with reduced bar width
    y_pos = np.arange(len(descriptions))
    bars = ax.barh(y_pos, scores, align='center', color=colors, height=bar_width)
    if bar_max == 'absolute':
        ax.set_xlim(0, max(scores))
        bar_scale_factor = 0.80
    else:  # 'auto'
        ax.set_xlim(0, 1)
        bar_scale_factor = 0.9
    ax.set_ylim(-0.5, len(descriptions) - 0.5)
    ax.invert_yaxis()

    # Remove the axes
    ax.axis('off')

    rightmost_limit = ax.get_xlim()[1]  # Get the rightmost edge of the plot

    # Scale all bars by the scaling factor
    for bar in bars:
        bar.set_width(bar.get_width() * bar_scale_factor)

    # Add the scores on the bars
    for i, (v, bar, color) in enumerate(zip(scores, bars, colors)):
        ax.text(rightmost_limit - 0.02, i + 0.05, f"{v:.3f}", color=color, va='center', ha='right', fontsize=fontsize)

    # Adjust font size based on the length of the longest description
    max_desc_length = max(map(len, descriptions))
    if max_desc_length > max_text_length:
        fontsize -= (max_desc_length - max_text_length) * 0.1

    # Ensure font size doesn't get too small
    fontsize = max(fontsize, 8)

    # Add the descriptions above the bars
    for i, desc in enumerate(descriptions):
        # Check for long descriptions and split them into two parts
        if len(desc) > max_text_length:
            split_index = desc.rfind(' ', 0, max_text_length)  # Find the last space before max_text_length
            if split_index == -1:  # No space found, split at max_text_length
                split_index = max_text_length
            part1 = desc[:split_index]
            part2 = desc[split_index + 1:]
            desc = part1 + '\n' + part2

        ax.text(0, i, desc, color='black', va='center', ha='left', fontsize=fontsize, multialignment='left')


html_response_style = f"""
<head>
    <title>AI Explanations</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
        }}
        .container {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: repeat(2, 1fr);
            gap: 20px;  /* Spacing between items */
        }}
        h3 {{
            margin: 5px 0;
        }}
        img {{
            max-width: 100%;
            height: auto;
        }}
        .explanation {{
            padding: 5px;
            box-sizing: border-box;
        }}
    </style>
</head>
"""

def generate_explanation(descriptions: list[str], scores: list[float], colors: list[str], exp_length: int = 500, sort_by_score: bool = True):
    MAX_LENGTH = 50
    fontsize = 15
    
    # compute the total number of lines we need, based on the length of the descriptions
    wrapped_lines = [textwrap.wrap(desc, width=MAX_LENGTH) for desc in descriptions]
    num_lines = sum(len(wrap) for wrap in wrapped_lines)
    height = num_lines * fontsize + 5 * len(descriptions) + 3

    # Sort descriptions, scores, and colors by scores in descending order
    combined = list(zip(scores, descriptions, colors))
    if sort_by_score:
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
    scores, descriptions, colors = zip(*combined)

    # Start the SVG inside a div
    svg_parts = [f'<div style="width: {exp_length}px; height: {height}px; background-color: white;">',
                 "<svg width=\"100%\" height=\"100%\">"]

    # Add a row for each description
    y_offset = 0
    bat_max_length = exp_length - 100
    for score, desc, color in zip(scores, descriptions, colors):

        # Calculate the length of the bar (scaled to fit within the SVG)
        bar_length = max(score, 0) * bat_max_length

        # Draw the bar
        y_offset += fontsize
        svg_parts.append(f"""
        <rect x="0" y="{y_offset-fontsize}" width="{bar_length}" height="{fontsize}" fill="{color}">
        </rect>
        """)
        
        # Overlay the description text on the bar
        wrapped_desc = textwrap.wrap(desc, width=MAX_LENGTH)
        for i, desc_line in enumerate(wrapped_desc):
            adjust = -fontsize/4 if i == 0 else fontsize*i
            svg_parts.append(f"""
            <text x="5" y="{y_offset + adjust}" font-size="{fontsize}" fill="black">
                {desc_line}
            </text>
            """)
            y_offset += adjust

        # Add the score
        svg_parts.append(f'<text x="{exp_length - 100}" y="{y_offset}" font-size="{fontsize}" fill={color} text-anchor="end">{score:.2f}</text>')

        y_offset += 8
        
    svg_parts.extend(("</svg>", "</div>"))
    # Join everything into a single string
    html = "".join(svg_parts)

    return html


def generate_html_layout(original_image: str, image_with_boxes: str, 
                         sachit_org_explanation: str, sachit_random_explanation: str, 
                         xclip_org_explanation: str, xclip_random_explanation: str, 
                         titles: list[str]) -> str:

    # Using CSS Grid for the layout
    html_content = f"""
    <body>
        <style>
            .container {{
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
            }}
            .image, .explanation {{
                width: 100%;
                box-sizing: border-box;
                overflow: hidden;  # To ensure content fits
            }}
        </style>
        
        <div class="container">
            <!-- Original Image -->
            <div class="image">
                <h3>{titles[0]}</h3>
                <img src="data:image/jpeg;base64,{original_image}" alt="Original Image">
            </div>
            
            <!-- M&V prediction with correct description -->
            <div class="explanation">
                <h3>{titles[1]}</h3>
                {sachit_org_explanation}
            </div>
            
            <!-- M&V prediction with nonsense description -->
            <div class="explanation">
                <h3>{titles[2]}</h3>
                {sachit_random_explanation}
            </div>
            
            <!-- Image with boxes -->
            <div class="image">
                <img src="data:image/jpeg;base64,{image_with_boxes}" alt="Boxed Image">
            </div>
            
            <!-- Our prediction with correct description -->
            <div class="explanation">
                <h3>{titles[4]}</h3>
                {xclip_org_explanation}
            </div>
            
            <!-- Our prediction with nonsense description -->
            <div class="explanation">
                <h3>{titles[5]}</h3>
                {xclip_random_explanation}
            </div>
        </div>
    </body>
    """

    return html_content



def img_to_base64(img):
    img_pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    buffered = io.BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str.decode()

def generate_org_vs_random(xclip_results: str = "",
                           xclip_random_results: str= "", 
                           sachit_results: str = "",
                           sachit_random_results: str = "",
                           out_folder: str = 'temp/cub_org_vs_random',
                           num_samples: int = None):
    file_name_as_key = lambda x: {item['file_name']: item for item in x}
    
    os.makedirs(out_folder, exist_ok=True)
    xclip_results = json.load(open(xclip_results, 'r'))
    xclip_random_results = json.load(open(xclip_random_results, 'r'))
    sachit_results = json.load(open(sachit_results, 'r'))
    sachit_random_results = json.load(open(sachit_random_results, 'r'))
    # use file name as key for the dictionary
    xclip_results = file_name_as_key(xclip_results)
    xclip_random_results = file_name_as_key(xclip_random_results)
    sachit_results = file_name_as_key(sachit_results)
    sachit_random_results = file_name_as_key(sachit_random_results)
    
    xclip_confusion_dict = results_confusion_matrix(xclip_results, xclip_random_results, methods=["xclip", "xclip_random"], file_name_only=True)
    sachit_confusion_dict = results_confusion_matrix(sachit_results, sachit_random_results, methods=["sachit", "sachit_random"], file_name_only=True)

    # we are intrested in TF in xclip and TT in sachit
    # TT sachit
    sachit_TT = sachit_confusion_dict['TT']
    # TF xclip
    xclip_TF = xclip_confusion_dict['TF']
    
    intersect_images = list(set(sachit_TT) & set(xclip_TF))
    if num_samples is not None:
        intersect_images = np.random.choice(intersect_images, num_samples, replace=False)
    
    for idx, file_name in tqdm(enumerate(intersect_images), desc="Generating explanations", total=len(intersect_images)):
        base_name = Path(file_name).stem
        sachit_result_org = sachit_results[file_name]
        sachit_result_random = sachit_random_results[file_name]
        xclip_results_org = xclip_results[file_name]
        xclip_results_random = xclip_random_results[file_name]
        org_image = Image.open(os.path.join('runs/explanations/cub_test/org', file_name)).convert('RGB')
        boxed_image = Image.open(os.path.join('runs/explanations/cub_test/boxes', f'{base_name}_all.jpg')).convert('RGB')
        
        # get sachit data
        sachit_org_desc = sachit_result_org['descriptions']
        sachit_org_scores = sachit_result_org['scores']
        sachit_random_desc = sachit_result_random['descriptions']
        sachit_random_scores = sachit_result_random['scores']
        sachit_colors_org = [SACHIT_COLOR] * len(sachit_org_desc)
        sachit_colors_random = [SACHIT_COLOR] * len(sachit_random_desc)
        
        # get xclip data
        xclip_org_desc = [xclip_results_org['descriptions'][part] for part in ORDERED_PARTS]
        xclip_org_scores = [max(xclip_results_org['pred_scores'][part], 0) for part in ORDERED_PARTS]
        xclip_random_desc = [xclip_results_random['descriptions'][part] for part in ORDERED_PARTS]
        xclip_random_scores = [max(xclip_results_random['pred_scores'][part], 0) for part in ORDERED_PARTS]
        xclip_colors = list(PART_COLORS.values())
        
        # get the labels
        gt_label = sachit_result_org['ground_truth']
        sachit_org_pred = sachit_result_org['pred']
        sachit_org_pred_score = np.mean(sachit_result_org['softmax_score'])
        sachit_random_pred = sachit_result_random['pred']
        sachit_random_pred_score = np.mean(sachit_result_random['softmax_score'])
        xclip_org_pred = xclip_results_org['pred']
        xclip_org_pred_score = np.mean(xclip_results_org['softmax_score'])
        xclip_random_pred = xclip_results_random['pred']
        xclip_random_pred_score = np.mean(xclip_results_random['softmax_score'])

        fontsize = 16
        label_size = 20
        # Desired total height for the figure
        desired_total_height = 10
        exp_image_width = 8

        # Calculate the desired height for each row
        row_height = desired_total_height / 2

        # Calculate the aspect ratio of the original image
        aspect_ratio = org_image.size[0] / org_image.size[1]
        
        # Calculate the width of the original image
        org_img_subplot_width = aspect_ratio * row_height

        # The total width of the figure will be the sum of the original image subplot width and two times the exp_image_width (since there are two columns for exp images)
        fig_width = org_img_subplot_width + 2 * exp_image_width

        # Create a master figure
        fig = plt.figure(figsize=(fig_width, desired_total_height), constrained_layout=True)

        print(f"org image width {org_img_subplot_width}, exp image width {exp_image_width}, total image width {fig_width}")
        # Create a 2x3 GridSpec layout
        gs = gridspec.GridSpec(2, 3, figure=fig, 
                            width_ratios=[org_img_subplot_width, exp_image_width, exp_image_width],
                            hspace=0.1, wspace=0.00,
                            )

        # Plot org_image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(org_image)
        # ax1.set_xlim(0, scaled_width)
        # ax1.set_ylim(0, row_height)
        ax1.axis('off')
        ax1.set_title(gt_label, fontsize=label_size, fontweight='bold')

        # Plot sachit_org_exp
        ax2 = fig.add_subplot(gs[0, 1])
        get_exp_overlay2(ax2, sachit_org_desc, sachit_org_scores, colors=sachit_colors_org, fontsize=fontsize)
        set_colored_title(ax2, sachit_org_pred, gt_label, sachit_org_pred_score, fontsize=fontsize)

        # Plot sachit_random_exp
        ax3 = fig.add_subplot(gs[0, 2])
        get_exp_overlay2(ax3, sachit_random_desc, sachit_random_scores, colors=sachit_colors_random, fontsize=fontsize)
        set_colored_title(ax3, sachit_random_pred, gt_label, sachit_random_pred_score, fontsize=fontsize)

        # Plot boxed_image
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.imshow(boxed_image)
        # ax4.set_xlim(0, scaled_width)
        # ax4.set_ylim(0, row_height)
        ax4.axis('off')
        ax4.set_title(" ")

        # Plot xclip_org_exp
        ax5 = fig.add_subplot(gs[1, 1])
        get_exp_overlay2(ax5, xclip_org_desc, xclip_org_scores, colors=xclip_colors, fontsize=fontsize)
        set_colored_title(ax5, xclip_org_pred, gt_label, xclip_org_pred_score, fontsize=fontsize)

        # Plot xclip_random_exp
        ax6 = fig.add_subplot(gs[1, 2])
        get_exp_overlay2(ax6, xclip_random_desc, xclip_random_scores, colors=xclip_colors, fontsize=fontsize)
        set_colored_title(ax6, xclip_random_pred, gt_label, xclip_random_pred_score, fontsize=fontsize)


        # Save the entire figure as a PDF
        output_path = os.path.join(out_folder, f"{base_name}.pdf")
        fig.savefig(output_path, format="pdf", bbox_inches='tight')
        
        
        plt.close(fig)


from PIL import Image, ImageDraw, ImageFont

def draw_iou_on_image(image, part_names, ious, colors, font_path: str = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
    """
    Draw IoU values on the image using PIL.
    
    :param image: The PIL image on which to draw.
    :param part_names: List of part names.
    :param ious: IoU values corresponding to each part.
    :param colors: Colors for each part.
    :param font_path: Path to the .ttf font file.
    :return: Modified image with IoU values drawn.
    """
    # Convert the image to RGB mode (if not already) to ensure we can draw on it
    image = image.convert("RGB")
    
    # Initialize ImageDraw
    draw = ImageDraw.Draw(image)
    
    # Use a truetype font with desired size
    font_size = 12
    font = ImageFont.truetype(font_path, font_size)
    
    # Starting y-coordinate
    y_offset = 5
    row_spacing = 2
    
    for part_name, iou, color in zip(part_names, ious, colors):
        text = f"{part_name}: {iou:.2f}"
        draw.text((5, y_offset), text, font=font, fill=color)
        y_offset += font_size + row_spacing  # Adjust this value based on your font size and desired spacing

    return image

def concatenate_images(*images):
    """
    Concatenate PIL Images horizontally.
    """
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    
    concatenated_img = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for img in images:
        concatenated_img.paste(img, (x_offset, 0))
        x_offset += img.width
        
    return concatenated_img

def draw_legend(image, part_names, colors, font_path):
    """
    Draw a compact legend on the image for each part name and color.
    """
    new_image = copy.deepcopy(image)
    draw = ImageDraw.Draw(new_image)
    font_size = 12  # Reduced font size
    font = ImageFont.truetype(font_path, font_size)
    
    box_size = 10  # Reduced size of the color box
    spacing = 3    # Reduced spacing
    
    x_offset, y_offset = 5, 5  # Starting position for the legend
    
    # Calculate the width and height for the background rectangle
    max_text_width = max(font.getlength(part_name) for part_name in part_names)
    total_width = x_offset + box_size + 5 + max_text_width + 5
    total_height = len(part_names) * (box_size + spacing) + spacing
    
    # Draw white background rectangle for the legend
    draw.rectangle([x_offset, y_offset, x_offset + total_width, y_offset + total_height], fill="white")
    
    # Update y_offset to start drawing inside the white rectangle
    y_offset += spacing
    
    for part_name, color in zip(part_names, colors):
        draw.rectangle([x_offset + 5, y_offset, x_offset + 5 + box_size, y_offset + box_size], fill=color)
        draw.text((x_offset + 5 + box_size + 5, y_offset), part_name, font=font, fill="black")
        y_offset += box_size + spacing  # Move to the next line

    return new_image

def generate_box_comparison(result_dict: str = 'runs/logits/xclip_birdsoupv2/final/cub_200_finetuned/xclip_pred.json', 
                            base_dict: str = 'runs/logits/xclip_birdsoupv2/final/owlvit_base_patch32_cub.json', 
                            teacher_folder: str = '/home/lab/xclip/owlvit_boxes/bird_soup/data',
                            out_folder: str = 'temp/cub_box_comparison', 
                            org_image_folder: str = 'runs/explanations/cub_test/org',
                            teacher_image_folder: str = 'runs/explanations/cub_test/boxes',
                            num_samples: int = None,):
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    result_dict = json.load(open(result_dict, 'r'))
    base_dict = json.load(open(base_dict, 'r'))
    base_dict = {item['file_name']: item for item in base_dict}
    
    if num_samples is not None:
        result_dict = np.random.choice(result_dict, num_samples, replace=False)
        
    
    os.makedirs(out_folder, exist_ok=True)
    
    for idx, item in tqdm(enumerate(result_dict), total=len(result_dict), desc="Drawing boxes"):
        file_name = item['file_name']
        file_name_base = Path(file_name).stem
        bs_file_name = item['bs_file_name']
        bs_file_name_base = Path(bs_file_name).stem
        
        pred_boxes = item['pred_boxes']
        pred_boxes = [pred_boxes[part] for part in ORDERED_PARTS]
        base_boxes = base_dict[file_name]['pred_boxes']
        base_boxes = [base_boxes[part] for part in ORDERED_PARTS]
        teacher_boxes = torch.load(os.path.join(teacher_folder, f'{bs_file_name_base}.pth'))
        teacher_boxes = teacher_boxes['boxes_info'] if 'boxes_info' in teacher_boxes else teacher_boxes['part_boxes']
        teacher_boxes = [teacher_boxes[part] for part in ORDERED_PARTS]
        
        org_image = Image.open(os.path.join(org_image_folder, file_name)).convert('RGB')
        teacher_image = Image.open(os.path.join(teacher_image_folder, f'{file_name_base}_all.jpg')).convert('RGB')

        image_size = org_image.size
        image_size = torch.tensor(image_size).repeat(len(teacher_boxes), 1)
        pred_boxes_norm = normalize_boxes(torch.tensor(pred_boxes), image_size)
        base_boxes_norm = normalize_boxes(torch.tensor(base_boxes), image_size)
        gt_boxes_norm = normalize_boxes(torch.tensor(teacher_boxes), image_size)

        # get giou, l1, iou for pred boxes
        giou_pred, iou_pred = compute_giou(pred_boxes_norm, gt_boxes_norm)
        # get the diagonal of the gious
        giou_pred = torch.diag(giou_pred)
        iou_pred = torch.diag(iou_pred)
        giou_loss_pred = 1 - giou_pred
        l1_loss_pred = compute_l1_loss(pred_boxes_norm, gt_boxes_norm)
        
        # get giou, l1, iou for base boxes
        giou_base, iou_base = compute_giou(base_boxes_norm, gt_boxes_norm)
        iou_base = torch.diag(iou_base)
        giou_base = torch.diag(giou_base)
        giou_loss_base = 1 - giou_base
        l1_loss_base = compute_l1_loss(base_boxes_norm, gt_boxes_norm)

        img_pred_boxes = Drawer.draw_boxes(org_image, pred_boxes, list(PART_COLORS_INT.values()), width=2)
        img_base_boxes = Drawer.draw_boxes(org_image, base_boxes, list(PART_COLORS_INT.values()), width=2)
        
        # write the iou on the image with format "part_name: iou" with list(PART_COLORS.values())
        img_pred_boxes_pil = draw_iou_on_image(img_pred_boxes, ORDERED_PARTS, iou_pred, list(PART_COLORS_INT.values()))
        img_base_boxes_pil = draw_iou_on_image(img_base_boxes, ORDERED_PARTS, iou_base, list(PART_COLORS_INT.values()))

        # concat the three images [org_image, image_pred_boxes_pil, img_base_boxes_pil, teacher_image], and add legend for the colors at the first image. 
        img_with_legend = draw_legend(org_image, ORDERED_PARTS, list(PART_COLORS_INT.values()), font_path)
        concatenated_image = concatenate_images(img_with_legend, img_pred_boxes_pil, img_base_boxes_pil, teacher_image)
        
        concatenated_image.save(os.path.join(out_folder, f'{file_name_base}.jpg'))

