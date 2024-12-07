# WSSS-ISIC
This is the repository for final project "Weakly Supervised Semantic Segmentation on Skin Lesion" of course CS 6357-01 Open-source Prog Med Img Anlys (2024F) instructed by Dr. Ipel Oguz.

## Environment set up
```
conda env create -f environment.yaml
```
Since our code are adapted and modified from https://github.com/linyq2117/CLIP-ES, we also encourage you to do the following:
```
git clone https://github.com/linyq2117/CLIP-ES.git
```

## Generate CAMs
You can generate CAMS by running:
```
python generate_cams_isic18.py --img_root /path/to/ISIC2018 --groundtruth /path/to/ISIC2018_Task3_Training_GroundTruth.csv --cam_out_dir ./output/isic/cams --model ./pretrained_models/clip/ViT-B-16.pt 
```

## Generate pseudo masks from CAMS
```
python eval_cam_with_crf.py --cam_out_dir ./output/isic/cams2/ --pseudo_mask_save_path ./output/isic/pseudo_masks/ --image_root /path/to/ISIC2018
```

## Train segmentation model with pseudo masks
```
python train_with_mask.py
```
