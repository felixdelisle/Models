# Models

## BRISC multiclass segmentation data

`segmentation_dataset.py` builds paired image/mask loaders for each anatomical
plane, using images from `classification_task` and corresponding masks from
`segmentation_task`. The output mask is a multiclass label map:

| Index | Tumor class |
|---:|---|
| 0 | Background (including no-tumor images) |
| 1 | Glioma |
| 2 | Meningioma |
| 3 | Pituitary |

```python
from segmentation_dataset import make_plane_dataloaders

loaders = make_plane_dataloaders("datasets/brisc2025")
axial_train = loaders["axial"]["train"]
coronal_train = loaders["coronal"]["train"]
sagittal_train = loaders["sagittal"]["train"]
```

Each plane loader supplies paired images and masks to its corresponding
encoder. Images and masks are resized to `256 x 256` by default; mask values are
integer class IDs suitable for a four-channel segmentation model (set
`num_classes=4`). `train.ipynb` trains a separate MobileUNETR for each plane
and defines `CustomLoss`, which combines cross-entropy with multiclass Tversky
loss; use it with targets shaped `[batch, height, width]`. No-tumor images are
included with empty (background-only) masks.
The dataset provides individual planes rather than aligned three-plane groups,
so the plane-specific loaders do not combine views into a single sample.

`MobileUNETR`, `AttentionUNet`, and `UNet` accept `multiple_encoders=True` to
enable three independent encoders. As in MobileUNETR, each branch applies
squeeze-and-excitation to all three skip tensors and its bottleneck, then the
branches use concatenation followed by learned `1x1` convolutions to fuse the
bottleneck and each skip. Pass the three matching views as a tuple:

```python
model = MobileUNETR(in_channels=3, num_classes=4, multiple_encoders=True)
logits = model((axial_image, coronal_image, sagittal_image))
```

Each input tensor must have the same batch and spatial dimensions. This
multi-view mode requires aligned axial/coronal/sagittal images for each sample;
the current BRISC files are not aligned into such triplets.

```python
criterion = CustomLoss().to(DEVICE)
images, masks = next(iter(axial_train))
single_view_model = MobileUNETR(in_channels=3, num_classes=4)
logits = single_view_model(images)  # [batch, 4, height, width]
loss = criterion(logits, masks)  # masks: [batch, height, width], integer IDs
```
