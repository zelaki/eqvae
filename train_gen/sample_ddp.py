
"""
Code adapted from https://github.com/chuanyangjin/fast-DiT
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../train_eqvae'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))


import torch
import torch.distributed as dist
from download import find_model
from models.dit_models import DiT_models
from models.diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm
import os
from PIL import Image
import numpy as np
import math
import argparse
from evaluator import Evaluator
import tensorflow.compat.v1 as tf
from train_eqvae.ldm.models.autoencoder import LDMAutoencoderKL
from ldm.util import instantiate_from_config
import yaml
from omegaconf import OmegaConf

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_kl(config, ckpt_path=None):

    model = LDMAutoencoderKL(**config.model.params)

    if ckpt_path is not None:
        sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()

def preprocess_kl(x):
    x = 2.*x - 1.
    return x



def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path



def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.)/2.
    x = x.permute(1,2,0).numpy()
    x = (255*x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def calculate_metrics(ref_batch, sample_batch, fid_path):
    config = tf.ConfigProto(
        allow_soft_placement=True  
    )
    config.gpu_options.allow_growth = True
    evaluator = Evaluator(tf.Session(config=config))

    evaluator.warmup()

    ref_acts = evaluator.read_activations(ref_batch)
    ref_stats, ref_stats_spatial = evaluator.read_statistics(ref_batch, ref_acts)

    sample_acts = evaluator.read_activations(sample_batch)
    sample_stats, sample_stats_spatial = evaluator.read_statistics(sample_batch, sample_acts)

    with open(fid_path, 'w') as fd:

        fd.write("Computing evaluations...\n")
        fd.write(f"Inception Score:{evaluator.compute_inception_score(sample_acts[0])}\n" )
        fd.write(f"FID:{sample_stats.frechet_distance(ref_stats)}\n")
        fd.write(f"sFID:{sample_stats_spatial.frechet_distance(ref_stats_spatial)}\n")
        prec, recall = evaluator.compute_prec_recall(ref_acts[0], sample_acts[0])
        fd.write(f"Precision:{prec}\n")
        fd.write(f"Recall:{recall}\n")



def main(args):
    """
    Run sampling.
    """
    torch.backends.cuda.matmul.allow_tf32 = args.tf32  # True: fast but may lead to some small numerical differences
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))

    if args.vae_ckpt is not None:
        vae_config = load_config(args.vae_config, display=False)
        vae = load_kl(vae_config, ckpt_path=args.vae_ckpt).to(device)
    else:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(args.hf_model_name).to(device)


    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"

    if args.ddpm:
        folder_name += f"-ddpm"


    sample_folder_dir = f"{args.sample_dir}/{folder_name}"

    exp_dir_name = os.path.dirname(args.sample_dir)
    fid_dir =  f"{exp_dir_name}/fids"

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(fid_dir, exist_ok=True)

        print(f"Saving .png samples at {sample_folder_dir}")
        print(f"Saving metrics at {fid_dir}")

    dist.barrier()

    fid_path = f"{fid_dir}/metrics-adm-{folder_name}.txt"

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0
    for _ in pbar:
        # Sample inputs:
        z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
        y = torch.randint(0, args.num_classes, (n,), device=device)

        # Setup classifier-free guidance:
        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            sample_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            sample_fn = model.forward

        # Sample images:
        if args.ddpm:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        else:
            samples = diffusion.ddim_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

        if args.vae_ckpt is not None:

            samples = vae.decode(samples / args.vae_scaling_factor)

        else:

            samples = vae.decode(samples / args.vae_scaling_factor).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Save samples to disk as individual .png files
        for i, sample in enumerate(samples):
            index = i * dist.get_world_size() + rank + total

            if args.vae in ["ours", "hf"]:
                sample = custom_to_pil(sample)
                sample.save(f"{sample_folder_dir}/{index:06d}.png")
            else:
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
        total += global_batch_size

    # Make sure all processes have finished saving their samples before attempting to convert to .npz
    dist.barrier()
    if rank == 0:
        sample_batch = create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        calculate_metrics(sample_batch, args.ref_batch, fid_path)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae-ckpt",  type=str,  default=None)
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--ddpm", type=bool, default=False)
    parser.add_argument("--vae-scaling-factor", type=float, default=0.18215)
    parser.add_argument("--ref-batch", type=str, default="/data/imagenet/VIRTUAL_imagenet256_labeled.npz")
    parser.add_argument("--wavelets", type=str, default=False)
    parser.add_argument("--diff-proj", type=str, default=False)
    parser.add_argument("--gaussian-registers", type=str, default=False)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
