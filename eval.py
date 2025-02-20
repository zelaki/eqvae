# Code modified from https://github.com/hustvl/LightningDiT/blob/main/evaluate_tokenizer.py

import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler
from evaluation.calculate_fid import calculate_fid_given_paths
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics import StructuralSimilarityIndexMeasure
from evaluation.lpips import LPIPS
from torchvision.datasets import ImageFolder
from torchvision import transforms
import csv
import sys
from ldm.models.autoencoder import AutoencoderKL 
from ldm.util import instantiate_from_config
import yaml
from omegaconf import OmegaConf

def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def load_kl(config, type="sd", ckpt_path=None):
    model = AutoencoderKL(**config.model.params)
    sd = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return model.eval()


def print_with_prefix(content, prefix='Tokenizer Evaluation', rank=0):
    if rank == 0:
        print(f"\033[34m[{prefix}]\033[0m {content}")

def save_image(image, filename):
    Image.fromarray(image).save(filename)



def evaluate_tokenizer(config_path, model_name, data_path, output_path, ckpt_path):
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')


    vae_config = load_config(config_path, display=False)
    model = load_kl(vae_config, ckpt_path=ckpt_path).to(device)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create dataset and dataloader
    dataset = ImageFolder(root=data_path, transform=transform)
    distributed_sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=local_rank)
    val_dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        sampler=distributed_sampler
    )

    folder_name = model_name    

    save_dir = os.path.join(output_path, folder_name, 'decoded_images')
    ref_path = os.path.join(output_path, folder_name, 'ref_images')
    metric_path = os.path.join(output_path, folder_name, 'metrics.csv')

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ref_path, exist_ok=True)

    if local_rank == 0:
        print_with_prefix(f"Output dir: {save_dir}")
        print_with_prefix(f"Reference dir: {ref_path}")

    # Save reference images if needed
    ref_png_files = [f for f in os.listdir(ref_path) if f.endswith('.png')]
    if len(ref_png_files) < 50000:
        total_samples = 0
        for batch in val_dataloader:
            images = batch[0].to(device)
            for j in range(images.size(0)):
                img = torch.clamp(127.5 * images[j] + 128.0, 0, 255).cpu().permute(1, 2, 0).numpy().astype(np.uint8)
                Image.fromarray(img).save(os.path.join(ref_path, f"ref_image_rank_{local_rank}_{total_samples}.png"))
                total_samples += 1
                if total_samples % 100 == 0 and local_rank == 0:
                    print_with_prefix(f"Rank {local_rank}, Saved {total_samples} reference images")
    dist.barrier()

    # Initialize metrics
    lpips_values = []
    ssim_values = []
    lpips = LPIPS().to(device).eval()
    ssim_metric = StructuralSimilarityIndexMeasure().to(device)

    # Generate reconstructions and compute metrics
    if local_rank == 0:
        print_with_prefix("Generating reconstructions...")
    all_indices = 0

    for batch in val_dataloader:
        images = batch[0].to(device)
        with torch.no_grad():
            latents = model.encode(images).sample().to(torch.float32)
            decoded_images_tensor = model.decode(latents)

        decoded_images = torch.clamp(127.5 * decoded_images_tensor + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        # Compute metrics
        lpips_values.append(lpips(decoded_images_tensor, images).mean())
        ssim_values.append(ssim_metric(decoded_images_tensor, images))
        
        # Save reconstructions
        for i, img in enumerate(decoded_images):
            save_image(img, os.path.join(save_dir, f"decoded_image_rank_{local_rank}_{all_indices + i}.png"))
            if (all_indices + i) % 100 == 0 and local_rank == 0:
                print_with_prefix(f"Rank {local_rank}, Processed {all_indices + i} images")
        all_indices += len(decoded_images)
    dist.barrier()

    # Aggregate metrics across GPUs
    lpips_values = torch.tensor(lpips_values).to(device)
    ssim_values = torch.tensor(ssim_values).to(device)
    dist.all_reduce(lpips_values, op=dist.ReduceOp.AVG)
    dist.all_reduce(ssim_values, op=dist.ReduceOp.AVG)
    
    avg_lpips = lpips_values.mean().item()
    avg_ssim = ssim_values.mean().item()

    if local_rank == 0:
        # Calculate FID
        print_with_prefix("Computing rFID...")
        fid = calculate_fid_given_paths([ref_path, save_dir], batch_size=50, dims=2048, device=device, num_workers=16)

        # Calculate PSNR
        print_with_prefix("Computing PSNR...")
        psnr_values = calculate_psnr_between_folders(ref_path, save_dir)
        avg_psnr = sum(psnr_values) / len(psnr_values)
        with open(metric_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["FID", f"{fid:.3f}"])
            writer.writerow(["PSNR", f"{avg_psnr:.3f}"])
            writer.writerow(["LPIPS", f"{avg_lpips:.3f}"])
            writer.writerow(["SSIM", f"{avg_ssim:.3f}"])

    dist.destroy_process_group()


def decode_to_images(model, z):
    with torch.no_grad():
        images = model.decode(z)
        images = torch.clamp(127.5 * images + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
    return images

def calculate_psnr(original, processed):
    mse = torch.mean((original - processed) ** 2)
    return 20 * torch.log10(255.0 / torch.sqrt(mse)).item()

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)

def calculate_psnr_for_pair(original_path, processed_path):
    return calculate_psnr(load_image(original_path), load_image(processed_path))

def calculate_psnr_between_folders(original_folder, processed_folder):
    original_files = sorted(os.listdir(original_folder))
    processed_files = sorted(os.listdir(processed_folder))

    if len(original_files) != len(processed_files):
        print("Warning: Mismatched number of images in folders")
        return []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(calculate_psnr_for_pair,
                          os.path.join(original_folder, orig),
                          os.path.join(processed_folder, proc))
            for orig, proc in zip(original_files, processed_files)
        ]
        return [future.result() for future in as_completed(futures)]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='configs/eqvae_config.yaml')
    parser.add_argument('--model_name', type=str, default='eq_vae')
    parser.add_argument('--ckpt_path', type=str)
    parser.add_argument('--data_path', type=str, default='/path/to/your/imagenet/ILSVRC2012_validation/data')
    parser.add_argument('--output_path', type=str, default='/path/to/your/output')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    evaluate_tokenizer(config_path=args.config_path, model_name=args.model_name, data_path=args.data_path, output_path=args.output_path, ckpt_path=args.ckpt_path)
