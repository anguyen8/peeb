import torch
from matplotlib import cm

def center_to_corners_format_torch(bboxes_center: "torch.Tensor") -> "torch.Tensor":
    center_x, center_y, width, height = bboxes_center.unbind(-1)
    bbox_corners = torch.stack(
        # top left x, top left y, bottom right x, bottom right y
        [(center_x - 0.5 * width), (center_y - 0.5 * height), (center_x + 0.5 * width), (center_y + 0.5 * height)],
        dim=-1,
    )
    return bbox_corners

def get_pre_define_colors(num_classes, cmap_set: list[str] = None, in_rgb: bool = True, is_float: bool = False):
    if cmap_set is None:
        cmap_set = ['tab20', 'tab20b', 'tab20c']

    colors = []
    for cmap in cmap_set:
        colors.extend(list(cm.get_cmap(cmap).colors))

    if in_rgb:
        new_colors = []
        for color in colors:
            if is_float:
                new_colors.append(tuple(color))
            else:
                new_colors.append(tuple(int(x * 255) for x in color))
        colors = new_colors

    if num_classes > len(colors):
        print(f"WARNING: {num_classes} classes are requested, but only {len(colors)} colors are available. Predefined colors will be reused.")
        colors *= num_classes // len(colors) + 1
    else:
        colors = colors[:num_classes]

    return colors


class Drawer:
    @staticmethod
    def draw_boxes(image, boxes, colors, tags=None, alpha=0.8, width=1, text_size=12, loc='above'):
        if tags is not None:
            try:
                font = ImageFont.truetype("/home/thang/Projects/factCheck/src/arial.ttf", text_size)
            except Exception:
                font = ImageFont.truetype("arial.ttf", text_size)
            if len(boxes) != len(tags):
                raise ValueError('boxes and tags must have same length')

        for idx, box in enumerate(boxes):
            # If there are duplicated boxes, slightly adjust x and y for better visualization
            if boxes.count(box) > 1:
                box[0] += np.random.randint(-10, 10) * 0.1
                box[1] += np.random.randint(-10, 10) * 0.1

            color_rgba = colors[idx] + (int(alpha * 255),)
            image = draw_box(image, box, color=color_rgba, width=width)

            if tags is not None:
                tag = tags[idx]
                draw = ImageDraw.Draw(image, 'RGBA')
                tag_width, tag_height = font.getmask(tag).size

                if loc == 'above':
                    textbb_loc = [box[0], box[1] - tag_height, box[0] + tag_width, box[1]]
                    text_loc = box[0], box[1] - tag_height
                else:
                    textbb_loc = [box[0], box[1], box[0] + tag_width, box[1] + tag_height]
                    text_loc = box[0], box[1]

                draw.rectangle(textbb_loc, fill=color_rgba)
                draw.text(text_loc, tag, fill='white', font=font)

        return image

    @staticmethod
    def draw_text(image, text_list): # draw text as extra box in image
        image = draw_text(image, text_list)
        return image
        
    @staticmethod
    def concat(target_image_list: PIL.Image, horizontal: bool = True) -> PIL.Image:
        widths, heights = [], []
        for image in target_image_list:
            width, height = image.size
            widths.append(width)
            heights.append(height)

        total_width, max_width = sum(widths), max(widths)
        total_height, max_height = sum(heights), max(heights)

        # num_imgs = len(target_image_list)
        cat_img = Image.new('RGB', (total_width, max_height), color=(255, 255, 255)) if horizontal else Image.new('RGB', (max_width, total_height), color=(255, 255, 255))
        for idx, img in enumerate(target_image_list):
            if horizontal:
                cat_img.paste(img, (sum(widths[:idx]), 0))
            else:
                cat_img.paste(img, (0, sum(heights[:idx])))
        return cat_img
    
    @staticmethod
    def paste_patch_to_image(box, img_patch, org_image):
        image = copy.deepcopy(org_image)
        image_width, image_height = image.size
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, image_width)
        y2 = min(y2, image_height)
        img_patch = img_patch.resize((x2-x1, y2-y1))
        image.paste(img_patch, (x1, y1))
        return image
    
    @staticmethod
    def concat_imgs_in_folders(dir1, dir2, dir3=None, out_dir=None, horizontal=True):
        print("Checking file name matching...")
        diff, intersection = Drawer.check_name_matching(dir1, dir2)
        img_list = intersection
        
        os.makedirs(out_dir, exist_ok=True), 
        for img_name in tqdm(img_list, total=len(img_list), desc='Concatenating images'):
            if img_name.endswith(".json"):
                continue

            img1 = Image.open(f'{dir1}/{img_name}')
            img2 = Image.open(f'{dir2}/{img_name}')
            if dir3 is not None:
                img3 = Image.open(f'{dir3}/{img_name}')
                cat_img = Drawer.concat([img1, img2, img3], horizontal=horizontal)
            else:
                cat_img = Drawer.concat([img1, img2], horizontal=horizontal)
            cat_img.save(f'{out_dir}/{img_name}', quality=100, subsampling=0)
    
    @staticmethod
    def check_name_matching(folder1, folder2):
        img_list1 = os.listdir(folder1)
        img_list2 = os.listdir(folder2)
        diff = set(img_list1) - set(img_list2)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder2}')
            print(diff)
        diff = set(img_list2) - set(img_list1)
        if len(diff) > 0:
            print(f'{len(diff)} images are not in {folder1}')
            print(diff)
        
        intersection = set(img_list1) & set(img_list2)
        return len(diff) == 0, intersection

    @staticmethod
    def draw_watermark(image: PIL.Image, text: str) -> PIL.Image:
        width, height = image.size
        watermark = generate_text_img(text, text_color=(255, 255, 255), background_color=(0, 0, 0), img_width=width, img_height=50)
        new_np = np.concatenate((np.array(image), np.array(watermark)), axis=0)
        new_image = Image.fromarray(new_np)

        return new_image