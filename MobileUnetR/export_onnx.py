import argparse
from pathlib import Path

import torch

try:
    from .mobile_unetr import MobileUNETR
except ImportError:
    from mobile_unetr import MobileUNETR


def export_checkpoint(checkpoint_path: Path, height: int, width: int) -> Path:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    model = MobileUNETR(in_channels=3, num_classes=1)
    model.load_state_dict(state_dict)
    model.eval()

    output_path = checkpoint_path.with_suffix(".onnx")
    sample = torch.zeros(1, 3, height, width)
    torch.onnx.export(
        model,
        sample,
        output_path,
        input_names=["image"],
        output_names=["mask"],
        dynamic_axes={"image": {0: "batch_size"}, "mask": {0: "batch_size"}},
        opset_version=17,
        dynamo=False,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert every PyTorch checkpoint in a directory to ONNX."
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "checkpoints",
        help="Directory containing .pt checkpoints (default: ./checkpoints).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Input image height (default: 512).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Input image width (default: 512).",
    )
    args = parser.parse_args()

    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive integers")
    if not args.checkpoints_dir.is_dir():
        parser.error(f"Checkpoint directory does not exist: {args.checkpoints_dir}")

    checkpoint_paths = sorted(args.checkpoints_dir.glob("*.pt"))
    if not checkpoint_paths:
        parser.error(f"No .pt checkpoints found in: {args.checkpoints_dir}")

    for checkpoint_path in checkpoint_paths:
        output_path = export_checkpoint(checkpoint_path, args.height, args.width)
        print(f"Exported {checkpoint_path} -> {output_path}")


if __name__ == "__main__":
    main()
