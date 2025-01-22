import argparse, os
import numpy as np
from PIL import Image



def resize(image_path, output_path, size):
    image = Image.open(image_path).convert("RGB").resize(size)
    return image
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indir",
        type=str,
        nargs="?",
        help="dir containing image",
        )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to save output",
        )
    parser.add_argument(
        "--resize",
        type=tuple,
        nargs="?",
        default=(512, 512),
        help="resize image",
        )
    opt = parser.parse_args()
    inImage = os.listdir(opt.indir)
    print("resize images in", opt.indir)
    print("output to", opt.outdir)
    os.makedirs(opt.outdir, exist_ok=True)
    for img in inImage:
        image = os.path.join(opt.indir, img)
        output = os.path.join(opt.outdir, img)
        resizedImg = resize(image, output, opt.resize)
        resizedImg.save(output)
    print("Resized images saved to", opt.outdir)
    
    