import argparse, os
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler
def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded
def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(Image.open(image).convert("RGB").resize((512, 512)))
    image = image.astype(np.float32)/255.0
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image)

    mask = np.array(Image.open(mask).convert("L").resize((512, 512)))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch
def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [Image.fromarray(img.astype(np.uint8)) for img in result]

def predict(sampler, image, mask, prompt, ddim_steps, num_samples, scale, seed):
    # image = np.array(Image.open(image).convert("RGB").resize((512, 512)))
    # image = image.astype(np.float32)/255.0
    # image = image[None].transpose(0,3,1,2)
    # image = torch.from_numpy(image)

    # image = input_image["image"].convert("RGB")
    # mask = input_image["mask"].convert("RGB")
    # image = pad_image(init_image) # resize to integer multiple of 32
    # mask = pad_image(init_mask) # resize to integer multiple of 32
    test = np.array(Image.open(image[0]).convert("RGB").resize((512, 512)))
    width, height = test.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=num_samples,
        h=height, w=width
    )

    return result

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config",
    )
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to model weights",
    )
    parser.add_argument(
        "--inmask",
        type=str,
        nargs="?",
        help="dir containing mask (`example.png` with same name as image)",
    )
    parser.add_argument(
        "--inimage",
        type=str,
        nargs="?",
        help="dir containing image (`example.png`)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    opt = parser.parse_args()
    masks = sorted(os.path.join(opt.inmask, f) for f in os.listdir(opt.inmask))
    images = sorted(os.path.join(opt.inimage, f) for f in os.listdir(opt.inimage))
    print(f"Found {len(masks)} inputs.")

    config = OmegaConf.load(opt.config)
    model = initialize_model(opt.config, opt.weights)
    os.makedirs(opt.outdir, exist_ok=True)
    results = predict(sampler= model,
                      image=images,
                      mask=masks,
                      prompt="", ddim_steps=opt.steps,
                      num_samples=len(masks),
                      scale=1.0,
                      seed=np.random.randint(0, 1000))
    results.save(opt.outdir)