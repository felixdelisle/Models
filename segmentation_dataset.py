from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SEGMENTATION_CLASSES = ("background", "glioma", "meningioma", "pituitary")
SEGMENTATION_CLASS_TO_INDEX = {
    name: index for index, name in enumerate(SEGMENTATION_CLASSES)
}
CLASSIFICATION_CLASSES = ("glioma", "meningioma", "no_tumor", "pituitary")
TUMOR_CODES = {
    "glioma": "gl",
    "meningioma": "me",
    "no_tumor": "no",
    "pituitary": "pi",
}
PLANE_CODES = {"ax": "axial", "co": "coronal", "sa": "sagittal"}
SPLITS = ("train", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class BRISCSegmentationDataset(Dataset):
    """Multiclass segmentation samples for one BRISC plane and split."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        plane: str,
        image_size: tuple[int, int] = (256, 256),
        transform: Callable[
            [Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]
        ]
        | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if plane not in PLANE_CODES.values():
            raise ValueError(
                f"plane must be one of {tuple(PLANE_CODES.values())}, got {plane!r}"
            )

        self.root = Path(root)
        self.split = split
        self.plane = plane
        self.image_size = image_size
        self.transform = transform
        self.classification_dir = self.root / "classification_task" / split
        self.mask_dir = self.root / "segmentation_task" / split / "masks"
        self.plane_code = next(
            code for code, name in PLANE_CODES.items() if name == plane
        )
        self.samples: list[tuple[Path, Path | None]] = []

        for tumor_name in CLASSIFICATION_CLASSES:
            class_dir = self.classification_dir / tumor_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Classification class directory not found: {class_dir}"
                )

            for image_path in sorted(class_dir.iterdir()):
                if (
                    not image_path.is_file()
                    or image_path.suffix.lower() not in IMAGE_SUFFIXES
                ):
                    continue

                parts = image_path.stem.split("_")
                if len(parts) != 6 or parts[4] not in PLANE_CODES:
                    raise ValueError(
                        f"Unexpected BRISC image filename: {image_path.name}"
                    )
                expected_code = TUMOR_CODES[tumor_name]
                if parts[3] != expected_code:
                    raise ValueError(
                        f"Filename tumor code does not match class folder: {image_path}"
                    )
                if parts[4] != self.plane_code:
                    continue

                mask_path = self.mask_dir / f"{image_path.stem}.png"
                if tumor_name == "no_tumor":
                    mask_path = None
                elif not mask_path.is_file():
                    raise FileNotFoundError(
                        f"Segmentation mask not found for image {image_path.name}: "
                        f"{mask_path}"
                    )

                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(
                f"No {plane} segmentation images found in {self.classification_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self.samples[index]
        height, width = self.image_size

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )

        target_class = SEGMENTATION_CLASS_TO_INDEX.get(
            image_path.parent.name, SEGMENTATION_CLASS_TO_INDEX["background"]
        )
        if mask_path is None:
            mask = Image.new("L", (width, height), color=0)
        else:
            with Image.open(mask_path) as mask_file:
                mask = mask_file.convert("L").resize(
                    (width, height), Image.Resampling.NEAREST
                )

        if self.transform is not None:
            image_tensor, mask_tensor = self.transform(image, mask)
            if mask_tensor.ndim == 3 and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.squeeze(0)
            return image_tensor, mask_tensor.long()

        image_array = np.asarray(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
        mask_array = np.asarray(mask, dtype=np.uint8)
        mask_tensor = torch.from_numpy(mask_array >= 128).long() * target_class

        return image_tensor, mask_tensor


def make_plane_dataloaders(
    root: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    image_size: tuple[int, int] = (256, 256),
    transform: Callable[
        [Image.Image, Image.Image], tuple[torch.Tensor, torch.Tensor]
    ]
    | None = None,
) -> dict[str, dict[str, DataLoader]]:
    """Build train/test segmentation loaders keyed by anatomical plane."""
    return {
        plane: {
            split: DataLoader(
                BRISCSegmentationDataset(
                    root,
                    split,
                    plane,
                    image_size=image_size,
                    transform=transform,
                ),
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=num_workers,
            )
            for split in SPLITS
        }
        for plane in PLANE_CODES.values()
    }
