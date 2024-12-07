import os
import pandas as pd
import numpy as np
import torch
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
    EnsureType,
    AsDiscrete,
)
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.data import DataLoader, CacheDataset
from monai.config import print_config
from monai.utils import set_determinism
from tqdm import tqdm

# Set deterministic training for reproducibility
set_determinism(seed=42)

# Print MONAI configuration
print_config()

# Define dataset paths
train_images = "/path/to/train/images"
train_masks = "/path/to/train/masks"
val_images = "/path/to/val/images"
val_masks = "/path/to/val/masks"
labels_csv = "/path/to/labels.csv"  # Path to the CSV file containing labels

# Load labels from CSV
def load_labels(csv_path):
    """
    Load labels from CSV and return a dictionary mapping image names to labels.
    """
    labels_df = pd.read_csv(csv_path)
    labels = {}
    for _, row in labels_df.iterrows():
        labels[row['image']] = row[1:].to_numpy(dtype=np.int64)  # Assuming label columns are one-hot encoded
    return labels

image_labels = load_labels(labels_csv)

# Load and preprocess data
def prepare_data(image_dir, mask_dir, labels):
    data_dicts = []
    for img in os.listdir(image_dir):
        img_name = os.path.splitext(img)[0]
        if img_name in labels:  # Ensure the label exists for the image
            data_dicts.append({
                "image": os.path.join(image_dir, img),
                "label": os.path.join(mask_dir, img.replace(".jpg", ".png")),
                "class_label": labels[img_name]
            })
    return data_dicts

train_files = prepare_data(train_images, train_masks, image_labels)
val_files = prepare_data(val_images, val_masks, image_labels)

# Define MONAI transforms
train_transforms = Compose(
    [
        LoadImage(keys=["image", "label"]),
        EnsureChannelFirst(keys=["image", "label"]),
        ScaleIntensity(keys=["image"]),
        Resize(keys=["image", "label"], spatial_size=(256, 256)),
        EnsureType(keys=["image", "label"]),
    ]
)

val_transforms = Compose(
    [
        LoadImage(keys=["image", "label"]),
        EnsureChannelFirst(keys=["image", "label"]),
        ScaleIntensity(keys=["image"]),
        Resize(keys=["image", "label"], spatial_size=(256, 256)),
        EnsureType(keys=["image", "label"]),
    ]
)

# Create datasets and loaders
train_ds = CacheDataset(data=train_files, transform=train_transforms, cache_rate=1.0)
val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_rate=1.0)
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=4)

# Define U-Net model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(
    dimensions=2,
    in_channels=1,
    out_channels=2,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

# Define loss, optimizer, and metrics
loss_function = DiceLoss(to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dice_metric = DiceMetric(include_background=False, reduction="mean")

# Training loop
max_epochs = 20
val_interval = 2
best_metric = -1
best_metric_epoch = -1
save_dir = "unet_model.pth"

for epoch in range(max_epochs):
    print(f"Epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in tqdm(train_loader):
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    epoch_loss /= step
    print(f"Average training loss: {epoch_loss:.4f}")

    # Validation
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            val_dice = 0
            val_steps = 0
            for val_data in val_loader:
                val_steps += 1
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                val_outputs = model(val_images)
                val_outputs = torch.softmax(val_outputs, dim=1)
                val_outputs = AsDiscrete(argmax=True)(val_outputs)
                dice_metric(y_pred=val_outputs, y=val_labels)
            metric = dice_metric.aggregate().item()
            dice_metric.reset()
            print(f"Validation Dice: {metric:.4f}")
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), save_dir)
                print(f"Model saved at epoch {epoch + 1}")
print(f"Best validation dice: {best_metric:.4f} at epoch {best_metric_epoch}")

# Testing the saved model (optional)
model.load_state_dict(torch.load(save_dir))
model.eval()
print("Testing the saved model on validation data...")
test_dice = DiceMetric(include_background=False, reduction="mean")
with torch.no_grad():
    for val_data in val_loader:
        val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
        val_outputs = model(val_images)
        val_outputs = torch.softmax(val_outputs, dim=1)
        val_outputs = AsDiscrete(argmax=True)(val_outputs)
        test_dice(y_pred=val_outputs, y=val_labels)
final_metric = test_dice.aggregate().item()
print(f"Final test dice: {final_metric:.4f}")
