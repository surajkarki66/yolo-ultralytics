import torch
import argparse


from pathlib import Path
from pytorch_nndct.apis import Inspector

parser = argparse.ArgumentParser(description="Inspect PyTorch model with NNDCT")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to the Ultralytics checkpoint (.pt)",
)
parser.add_argument(
    "--target",
    type=str,
    default="DPUCZDX8G_ISA1_B4096",
    help="DPU target string (must match quantization and compilation)",
)
parser.add_argument(
    "--img_height",
    type=int,
    default=416,
    help="Input image height",
)
parser.add_argument(
    "--img_width",
    type=int,
    default=416,
    help="Input image width",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="inspect",
    help="Directory for inspector PNG output",
)
parser.add_argument(
    "--end2end",
    action="store_true",
    default=False,
    help="Inspect fused one2one export (default: one2many for CPU NMS)",
)
args = parser.parse_args()


def prepare_model_for_inspect(model):
    """Match vai_q_yolo.py export settings so inspection reflects the quantized graph."""
    head = model.model[-1]
    if not hasattr(head, "export"):
        print("[WARN] Head has no export flag; assuming stock Ultralytics head.")
        return model

    head.export = True
    print("[INFO] Head export=True (raw DPU logits)")

    if args.end2end and getattr(head, "end2end") and hasattr(head, "fuse"):
        head.fuse()
        print("[INFO] --end2end: fused one2many away; exporting one2one.")
    elif args.end2end:
        print("[INFO] --end2end requested but head has no one2one branch; exporting as-is.")
    elif hasattr(head, "end2end"):
        head.end2end = False
        print("[INFO] Default one2many export for CPU NMS.")
    return model

model_path = Path(args.model_path)
if not model_path.is_file():
    raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

inspector = Inspector(args.target)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Model: {model_path}")
print(f"Input shape: 1 3 {args.img_height} {args.img_width}")
print(f"Target: {args.target}")

checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
model = checkpoint["model"]
model = model.float()
model.eval()
model = prepare_model_for_inspect(model)
model = model.to(device)

dummy_input = torch.randn(1, 3, args.img_height, args.img_width, device=device)

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

inspector.inspect(
    model,
    (dummy_input,),
    device=device,
    output_dir=str(output_dir),
    image_format="png",
)

print(f"Inspection complete. Output saved to {output_dir}/")
