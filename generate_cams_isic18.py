# -*- coding:UTF-8 -*-
from pytorch_grad_cam import GradCAM
import torch
import clip
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd

from tqdm import tqdm
from pytorch_grad_cam.utils.image import scale_cam_image
from utils import parse_xml_to_dict, scoremap2bbox
from clip_text import class_names, full_class_names_ISIC, BACKGROUND_CATEGORY_ISIC#, imagenet_templates
import argparse
from lxml import etree
import time
from torch import multiprocessing
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import warnings
warnings.filterwarnings("ignore")
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0
import argparse
from transformers import CLIPProcessor, CLIPModel

from plip.plip import PLIP
# ISIC 2018 category names and their full-text descriptions
class_names = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
full_class_names = [
    "melanoma",
    "melanocytic nevus",
    "basal cell carcinoma",
    "actinic keratosis",
    "benign keratosis-like lesion",
    "dermatofibroma",
    "vascular lesion"
]

def reshape_transform(tensor, height=28, width=28):
    tensor = tensor.permute(1, 0, 2)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return result.transpose(2, 3).transpose(1, 2)


def split_dataset(label_map, n_splits):
    """
    Split the dataset (label_map) into n_splits parts for parallel processing.

    Args:
        label_map (dict): Mapping of image names to label indices.
        n_splits (int): Number of splits.

    Returns:
        list of dict: A list where each element is a dictionary containing a portion of the label_map.
    """
    if n_splits == 1:
        return [label_map]

    # Convert label_map keys to a list for slicing
    image_keys = list(label_map.keys())
    part_size = len(image_keys) // n_splits
    dataset_splits = []

    # Slice the keys and create split dictionaries
    for i in range(n_splits - 1):
        split_keys = image_keys[i * part_size: (i + 1) * part_size]
        dataset_splits.append({key: label_map[key] for key in split_keys})

    # Add the remaining images to the last split
    split_keys = image_keys[(n_splits - 1) * part_size:]
    dataset_splits.append({key: label_map[key] for key in split_keys})

    return dataset_splits

def _transform_resize(h, w):
    return Compose([
        Resize((h,w), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
def zeroshot_classifier(classnames, templates, model):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).to(device)
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
    return zeroshot_weights.t()

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def load_groundtruth(groundtruth_file):
    """Load GroundTruth CSV and return a mapping of image to label index."""
    df = pd.read_csv(groundtruth_file)
    label_map = {}
    for _, row in df.iterrows():
        image_name = row['image'] + '.jpg'  # Append extension to image name
        label_index = np.argmax(row[1:].values)  # Convert one-hot to index
        label_map[image_name] = label_index
    return label_map

def preprocess_image(image_path):
    """Preprocess the image for CLIP."""
    preprocess = Compose([
        Resize((224, 224)),  # Resize to CLIP input size
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    image = Image.open(image_path).convert("RGB")
    return preprocess(image).unsqueeze(0)

def img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0], patch_size=16):
    all_imgs = []
    for scale in scales:
        preprocess = _transform_resize(int(np.ceil(scale * int(ori_height) / patch_size) * patch_size), int(np.ceil(scale * int(ori_width) / patch_size) * patch_size))
        image = preprocess(Image.open(img_path))
        image_ori = image
        image_flip = torch.flip(image, [-1])
        all_imgs.append(image_ori)
        all_imgs.append(image_flip)
    return all_imgs

class ClipOutputTarget:
    def __init__(self, category):
        self.category = category
    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


# def perform(process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam):
#     """
#     Perform CAM generation for the ISIC 2018 dataset.
#
#     Args:
#         process_id (int): The ID of the current process.
#         dataset_list (list): Split dataset list for parallel processing.
#         args: Command-line arguments.
#         model: Loaded CLIP model.
#         bg_text_features: Background text features for Grad-CAM.
#         fg_text_features: Foreground text features for Grad-CAM.
#         cam: Grad-CAM instance.
#     """
#     # Define the mapping from class abbreviations to full names
#     abbrev_to_full = dict(zip(class_names, full_class_names))
#     # plip = PLIP('vinid/plip')
#
#     # Identify GPU for current process
#     n_gpus = torch.cuda.device_count()
#     device_id = "cuda:{}".format(process_id % n_gpus)
#     databin = dataset_list[process_id]
#     model = model.to(device_id)
#     bg_text_features = bg_text_features.to(device_id)
#     fg_text_features = fg_text_features.to(device_id)
#
#     for im_idx, (image_name, label_index) in enumerate(tqdm(databin.items())):
#         # Step 1: Construct image path
#         img_path = os.path.join(args.img_root, image_name)
#         if not os.path.exists(img_path):
#             print(f"Image {img_path} does not exist. Skipping.")
#             continue
#
#         # Step 2: Get class label
#         label_abbrev = class_names[label_index]  # Get abbreviation
#         label_full = abbrev_to_full[label_abbrev]  # Map to full name
#         label_list = [label_full]  # Use full name for compatibility
#         label_id_list = [label_index]
#
#         # Step 3: Preprocess image (multi-scale and flipping)
#         try:
#             ori_height = 600
#             ori_width = 450
#             ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
#             ms_imgs = [ms_imgs[0]]
#         except Exception as e:
#             print(f"Error in preprocessing image {image_name}: {e}")
#             continue
#
#         if len(label_list) == 0:
#             print(f"{image_name} has no valid labels. Skipping.")
#             continue
#
#         # ori_height = 600
#         # ori_width = 450
#         # ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
#         # ms_imgs = [ms_imgs[0]]
#         cam_all_scales = []
#         highres_cam_all_scales = []
#         refined_cam_all_scales = []
#         for image in ms_imgs:
#             image = image.unsqueeze(0)
#             h, w = image.shape[-2], image.shape[-1]
#             image = image.to(device_id)
#             image_features, attn_weight_list = model.encode_image(image, h, w)
#
#             cam_to_save = []
#             highres_cam_to_save = []
#             refined_cam_to_save = []
#             keys = []
#
#             bg_features_temp = bg_text_features.to(device_id)  # [bg_id_for_each_image[im_idx]].to(device_id)
#             fg_features_temp = fg_text_features[label_id_list].to(device_id)
#             text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
#             input_tensor = [image_features, text_features_temp.to(device_id), h, w]
#
#             for idx, label in enumerate(label_list):
#                 keys.append(full_class_names_ISIC.index(label))
#                 targets = [ClipOutputTarget(label_list.index(label))]
#
#                 # torch.cuda.empty_cache()
#                 grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
#                                                                         targets=targets,
#                                                                         target_size=None)  # (ori_width, ori_height))
#
#                 grayscale_cam = grayscale_cam[0, :]
#
#                 grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
#                 highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))
#
#                 if idx == 0:
#                     attn_weight_list.append(attn_weight_last)
#                     attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
#                     attn_weight = torch.stack(attn_weight, dim=0)[-8:]
#                     attn_weight = torch.mean(attn_weight, dim=0)
#                     attn_weight = attn_weight[0].cpu().detach()
#                 attn_weight = attn_weight.float()
#
#                 box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
#                 aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
#                 for i_ in range(cnt):
#                     x0_, y0_, x1_, y1_ = box[i_]
#                     aff_mask[y0_:y1_, x0_:x1_] = 1
#
#                 aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
#                 aff_mat = attn_weight
#
#                 trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
#                 trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
#
#                 for _ in range(2):
#                     trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
#                     trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
#                 trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2
#
#                 for _ in range(1):
#                     trans_mat = torch.matmul(trans_mat, trans_mat)
#
#                 trans_mat = trans_mat * aff_mask
#
#                 cam_to_refine = torch.FloatTensor(grayscale_cam)
#                 cam_to_refine = cam_to_refine.view(-1, 1)
#
#                 # (n,n) * (n,1)->(n,1)
#                 cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
#                 cam_refined = cam_refined.cpu().numpy().astype(np.float32)
#                 cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
#                 refined_cam_to_save.append(torch.tensor(cam_refined_highres))
#
#             keys = torch.tensor(keys)
#             # cam_all_scales.append(torch.stack(cam_to_save,dim=0))
#             highres_cam_all_scales.append(torch.stack(highres_cam_to_save, dim=0))
#             refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))
#
#         # cam_all_scales = cam_all_scales[0]
#         highres_cam_all_scales = highres_cam_all_scales[0]
#         refined_cam_all_scales = refined_cam_all_scales[0]
#
#         np.save(os.path.join(args.cam_out_dir, image_name.replace('.jpg', 'npy')),
#                 {"keys": keys.numpy(),
#                  # "strided_cam": cam_per_scales.cpu().numpy(),
#                  # "highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
#                  "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
#                  })
#     return 0
def perform(process_id, dataset_list, args, model, bg_text_features, fg_text_features, cam):
    """
    Perform CAM generation for the ISIC 2018 dataset.

    Args:
        process_id (int): The ID of the current process.
        dataset_list (list): Split dataset list for parallel processing.
        args: Command-line arguments.
        model: Loaded CLIP model.
        bg_text_features: Background text features for Grad-CAM.
        fg_text_features: Foreground text features for Grad-CAM.
        cam: Grad-CAM instance.
    """
    # Define the mapping from class abbreviations to full names
    abbrev_to_full = dict(zip(class_names, full_class_names))

    # Identify GPU for current process
    n_gpus = torch.cuda.device_count()
    device_id = "cuda:{}".format(process_id % n_gpus)
    databin = dataset_list[process_id]
    model = model.to(device_id)
    bg_text_features = bg_text_features.to(device_id)
    fg_text_features = fg_text_features.to(device_id)

    for im_idx, (image_name, label_index) in enumerate(tqdm(databin.items())):
        # Step 1: Construct image path
        img_path = os.path.join(args.img_root, image_name)
        if not os.path.exists(img_path):
            print(f"Image {img_path} does not exist. Skipping.")
            continue

        # Step 2: Get class label
        label_abbrev = class_names[label_index]  # Get abbreviation
        label_full = abbrev_to_full[label_abbrev]  # Map to full name
        label_list = [label_full]  # Use full name for compatibility
        label_id_list = [label_index]

        # Step 3: Preprocess image (multi-scale and flipping)
        try:
            ori_height = 600
            ori_width = 450
            ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
            ms_imgs = [ms_imgs[0]]
        except Exception as e:
            print(f"Error in preprocessing image {image_name}: {e}")
            continue

        # Step 4: Skip invalid labels
        if len(label_list) == 0:
            print(f"{image_name} has no valid labels. Skipping.")
            continue

        # Step 5: (Unchanged) Compute Grad-CAM and save results
        ori_height = 600
        ori_width = 450
        ms_imgs = img_ms_and_flip(img_path, ori_height, ori_width, scales=[1.0])
        ms_imgs = [ms_imgs[0]]
        cam_all_scales = []
        highres_cam_all_scales = []
        refined_cam_all_scales = []
        for image in ms_imgs:
            image = image.unsqueeze(0)
            h, w = image.shape[-2], image.shape[-1]
            image = image.to(device_id)
            image_features, attn_weight_list = model.encode_image(image, h, w)

            cam_to_save = []
            highres_cam_to_save = []
            refined_cam_to_save = []
            keys = []

            bg_features_temp = bg_text_features.to(device_id)  # [bg_id_for_each_image[im_idx]].to(device_id)
            fg_features_temp = fg_text_features[label_id_list].to(device_id)
            text_features_temp = torch.cat([fg_features_temp, bg_features_temp], dim=0)
            input_tensor = [image_features, text_features_temp.to(device_id), h, w]

            for idx, label in enumerate(label_list):
                keys.append(full_class_names_ISIC.index(label))
                targets = [ClipOutputTarget(label_list.index(label))]

                # torch.cuda.empty_cache()
                grayscale_cam, logits_per_image, attn_weight_last = cam(input_tensor=input_tensor,
                                                                        targets=targets,
                                                                        target_size=None)  # (ori_width, ori_height))

                grayscale_cam = grayscale_cam[0, :]

                grayscale_cam_highres = cv2.resize(grayscale_cam, (ori_width, ori_height))
                highres_cam_to_save.append(torch.tensor(grayscale_cam_highres))

                if idx == 0:
                    attn_weight_list.append(attn_weight_last)
                    attn_weight = [aw[:, 1:, 1:] for aw in attn_weight_list]  # (b, hxw, hxw)
                    attn_weight = torch.stack(attn_weight, dim=0)[-8:]
                    attn_weight = torch.mean(attn_weight, dim=0)
                    attn_weight = attn_weight[0].cpu().detach()
                attn_weight = attn_weight.float()

                box, cnt = scoremap2bbox(scoremap=grayscale_cam, threshold=0.4, multi_contour_eval=True)
                aff_mask = torch.zeros((grayscale_cam.shape[0], grayscale_cam.shape[1]))
                for i_ in range(cnt):
                    x0_, y0_, x1_, y1_ = box[i_]
                    aff_mask[y0_:y1_, x0_:x1_] = 1

                aff_mask = aff_mask.view(1, grayscale_cam.shape[0] * grayscale_cam.shape[1])
                aff_mat = attn_weight

                trans_mat = aff_mat / torch.sum(aff_mat, dim=0, keepdim=True)
                trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)

                for _ in range(2):
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=0, keepdim=True)
                    trans_mat = trans_mat / torch.sum(trans_mat, dim=1, keepdim=True)
                trans_mat = (trans_mat + trans_mat.transpose(1, 0)) / 2

                for _ in range(1):
                    trans_mat = torch.matmul(trans_mat, trans_mat)

                trans_mat = trans_mat * aff_mask

                cam_to_refine = torch.FloatTensor(grayscale_cam)
                cam_to_refine = cam_to_refine.view(-1, 1)

                # (n,n) * (n,1)->(n,1)
                cam_refined = torch.matmul(trans_mat, cam_to_refine).reshape(h // 16, w // 16)
                cam_refined = cam_refined.cpu().numpy().astype(np.float32)
                cam_refined_highres = scale_cam_image([cam_refined], (ori_width, ori_height))[0]
                refined_cam_to_save.append(torch.tensor(cam_refined_highres))

            keys = torch.tensor(keys)
            # cam_all_scales.append(torch.stack(cam_to_save,dim=0))
            highres_cam_all_scales.append(torch.stack(highres_cam_to_save, dim=0))
            refined_cam_all_scales.append(torch.stack(refined_cam_to_save, dim=0))

        # cam_all_scales = cam_all_scales[0]
        highres_cam_all_scales = highres_cam_all_scales[0]
        refined_cam_all_scales = refined_cam_all_scales[0]

        np.save(os.path.join(args.cam_out_dir, image_name.replace('.jpg', 'npy')),
                {"keys": keys.numpy(),
                 # "strided_cam": cam_per_scales.cpu().numpy(),
                 # "highres": highres_cam_all_scales.cpu().numpy().astype(np.float16),
                 "attn_highres": refined_cam_all_scales.cpu().numpy().astype(np.float16),
                 })
    return 0



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate CAM for ISIC 2018 Dataset using CLIP')
    parser.add_argument('--img_root', type=str, required=True, help='Path to the ISIC 2018 image directory')
    parser.add_argument('--groundtruth', type=str, required=True, help='Path to the GroundTruth CSV file')
    parser.add_argument('--cam_out_dir', type=str, required=True, help='Directory to save CAM results')
    parser.add_argument('--model', type=str, default='ViT-B/16', help='CLIP model variant')
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    # Load GroundTruth labels
    label_map = load_groundtruth(args.groundtruth)
    if not os.path.exists(args.cam_out_dir):
        os.makedirs(args.cam_out_dir)
    # Load CLIP model
    model, _ = clip.load(args.model, device=device)
    # model = CLIPModel.from_pretrained("vinid/plip")
    # processor = CLIPProcessor.from_pretrained("vinid/plip")
    # Generate text features for categories
    # templates = ["a dermoscopic image of {}.", "an example of {}."]
    # text_features = zeroshot_classifier(full_class_names, templates, model, device)
    bg_text_features = zeroshot_classifier(BACKGROUND_CATEGORY_ISIC, ['a dermoscopic image of {}.'], model)
    fg_text_features = zeroshot_classifier(full_class_names_ISIC, ['a dermoscopic image of {}.'], model)

    # Grad-CAM setup
    target_layers = [model.visual.transformer.resblocks[-1].ln_1]
    cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

    # List all images
    # dataset_list = [img for img in os.listdir(args.img_root) if img.endswith('.jpg')]
    dataset_list = split_dataset(label_map, n_splits=args.num_workers)

    # Perform CAM generation
    if args.num_workers == 1:
        perform(0, dataset_list, args, model, bg_text_features, fg_text_features, cam)
    else:
        multiprocessing.spawn(perform, nprocs=args.num_workers,
                              args=(dataset_list, args, model, bg_text_features, fg_text_features, cam))