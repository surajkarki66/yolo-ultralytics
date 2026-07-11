import os
import argparse
import random

from pytorch_nndct.apis import torch_quantizer
import torch
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
import cv2


import glob
import math
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()

parser.add_argument(
    '--model_path',
    default="best.pt",
    help='Ultralytics checkpoint path (e.g. best.pt or yolo26n.pt)',
)

parser.add_argument(
    '--config_file',
    default=None,
    help='quantization configuration file')

parser.add_argument(
    '--batch_size',
    default=32,
    type=int,
    help='input data batch size to evaluate model')


parser.add_argument('--quant_mode', 
    default='calib', 
    choices=['float', 'calib', 'test'], 
    help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')


parser.add_argument('--deploy', 
    dest='deploy',
    action='store_true',
    help='export xmodel for deployment')

parser.add_argument('--target', 
    dest='target',
    nargs="?",
    const="",
    help='specify target device')
    
parser.add_argument(
    '--img_height',
    default=640,
    type=int,
    help='Input image height'
)

parser.add_argument(
    '--img_width',
    default=640,
    type=int,
    help='Input image width'
)

parser.add_argument(
    '--end2end',
    action='store_true',
    default=False,
    help='Fuse one2many and export one2one for NMS-free top-k (default: one2many + CPU NMS)',
)


args, _ = parser.parse_known_args()


def detect_config_pickle_path(model_path: str) -> Path:
    """Build a stable pickle path from the checkpoint filename (e.g. best.pt -> best_...pkl)."""
    stem = Path(model_path).stem or "model"
    return Path("quantize_result") / f"{stem}_config_no_srd_reg_nc_dfl.pkl"


def run_model_exports(model):
    import pickle
    head = model.model[-1]
    tensor_no = head.no
    tensor_stride = head.stride
    tensor_reg_max = head.reg_max
    tensor_nc = head.nc
    layer_dfl = head.dfl
    end2end = getattr(head, "end2end", False)
    output_path = detect_config_pickle_path(args.model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump((tensor_no, tensor_stride, tensor_reg_max, tensor_nc, layer_dfl, end2end), f)
    print(f"Saved detect head config to {output_path}")
    return


def prepare_model_for_quant(model):
    """Configure detect head for DPU export; fuse one2many when end2end export is requested."""
    head = model.model[-1]
    if not hasattr(head, "export"):
        print("[WARN] Head has no export flag; assuming stock Ultralytics head.")
        return model

    head.export = True
    print("[INFO] Head export=True (raw DPU logits)")

    if args.end2end and getattr(head, "end2end") and hasattr(head, "fuse"):
        head.fuse()
        print("[INFO] --end2end: fused one2many away; exporting one2one for CPU top-k.")
    elif args.end2end:
        print("[INFO] --end2end requested but head has no one2one branch; exporting as-is.")
    elif hasattr(head, "end2end"):
        head.end2end = False
        print("[INFO] Default one2many export for CPU NMS.")
    return model


IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}  # image suffixes
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}"

class LoadImages:
    """
    Calibration image dataloader for YOLOv26 / YOLOv11 quantization.

    Loads image paths from a txt file, directory, glob, or single file and
    resizes each image to the target width and height before batching.
    """

    def __init__(self, path, batch=1, img_size=(640, 640)):
        """Initialize the dataloader and raise FileNotFoundError if file not found.

        Args:
            path: Image path, directory, glob, or txt list of paths.
            batch: Batch size.
            img_size: Target (width, height) for cv2.resize.
        """
        self.img_size = img_size
        parent = None
        if isinstance(path, str) and Path(path).suffix == ".txt":  # *.txt file with img/vid/dir on each line
            parent = Path(path).parent
            path = Path(path).read_text().splitlines()  # list of sources
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            a = str(Path(p).absolute())  # do not use .resolve() https://github.com/ultralytics/ultralytics/issues/2912
            if "*" in a:
                files.extend(sorted(glob.glob(a, recursive=True)))  # glob
            elif os.path.isdir(a):
                files.extend(sorted(glob.glob(os.path.join(a, "*.*"))))  # dir
            elif os.path.isfile(a):
                files.append(a)  # files (absolute or relative to CWD)
            elif parent and (parent / p).is_file():
                files.append(str((parent / p).absolute()))  # files (relative to *.txt file parent)
            else:
                raise FileNotFoundError(f"{p} does not exist")

        # Define files as images
        images = []
        for f in files:
            suffix = f.split(".")[-1].lower()  # Get file extension without the dot and lowercase
            if suffix in IMG_FORMATS:
                images.append(f)

        self.files = images
        self.nf = len(images)  # number of files
        self.mode = "image"
        self.bs = batch
        if self.nf == 0:
            raise FileNotFoundError(f"No images found. {FORMATS_HELP_MSG}")

    def __iter__(self):
        """Returns an iterator object for ImageFolder."""
        self.count = 0
        return self

    def __next__(self):
        """Returns the next batch of images along with their paths and metadata."""
        paths, imgs, info = [], [], []
        while len(imgs) < self.bs:
            if self.count >= self.nf:  # end of file list
                if len(imgs) > 0:
                    return paths, imgs, info  # return last partial batch
                else:
                    raise StopIteration

            path = self.files[self.count]
            im0 = cv2.imread(path)  # BGR
            if im0 is None:
                raise FileNotFoundError(f"Image Not Found {path}")
            im0 = cv2.resize(im0, self.img_size, interpolation=cv2.INTER_LINEAR)
            paths.append(path)
            imgs.append(im0)
            info.append(f"image {self.count + 1}/{self.nf} {path}: ")
            self.count += 1  # move to the next file

        return paths, imgs, info

    def __len__(self):
        """Returns the number of batches in the object."""
        return math.ceil(self.nf / self.bs)  # number of files

def experimental(model):
    img_size = (args.img_width, args.img_height)
    dataset = LoadImages("data/val_ids.txt", args.batch_size, img_size=img_size)
    for _, batch in tqdm(enumerate(dataset), total=len(dataset)):
        # processing
        _, transform_im, _ = batch
        batch_tensor = torch.stack([torch.from_numpy(data.transpose(2,0,1)).to(device).float() / 255.0 for data in transform_im])

        _ = model(batch_tensor)
    print("Done")

def quantization():
    print("quantization")
    quant_mode = args.quant_mode
    deploy = args.deploy
    batch_size = args.batch_size
    config_file = args.config_file
    target = args.target

    checkpoint = torch.load(args.model_path, map_location=torch.device("cpu"))
    model = checkpoint["model"]
    
    model.float()
    model.eval()
    model = prepare_model_for_quant(model)

    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and batch_size != 1:
        print(r'Warning: Exporting xmodel needs batch size to be 1, changing it automatically!')
        batch_size = 1

    input = torch.randn([batch_size, 3, args.img_height, args.img_width])

    ####################################################################################
    # This function call will create a quantizer object and setup it. 
    # Eager mode model code will be converted to graph model. 
    # Quantization is not done here if it needs calibration.
    quantizer = torch_quantizer(quant_mode, model, (input), device=device, quant_config_file=config_file, target=target)
    print("quantizer done")

    if quant_mode == 'calib':
        print("Exporting model configuration...")
        run_model_exports(model)
        print("Done exporting configuration")
    # Get the converted model to be quantized.
    quant_model = quantizer.quant_model
    #####################################################################################

    # This function call is to do forward loop for model to be quantized.
    # Quantization calibration will be done after it if quant_mode is 'calib'.
    # Quantization test  will be done after it if quant_mode is 'test'.
    _ = experimental(quant_model)

    # handle quantization result
    if quant_mode == 'calib':
    # Exporting intermediate files will be used when quant_mode is 'test'. This is must.
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_torch_script()
        quantizer.export_onnx_model()
        quantizer.export_xmodel()


if __name__ == '__main__':
    print("-------- Start {} test --------".format(args.model_path))
    quantization()
    print("-------- End of {} test --------".format(args.model_path))
