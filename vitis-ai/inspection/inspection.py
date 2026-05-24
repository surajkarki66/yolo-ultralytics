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
args = parser.parse_args()

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
