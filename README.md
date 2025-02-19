<!--             
<style>
  .texttt {
    font-family: Consolas; /* Monospace font */
    font-size: 1em; /* Match surrounding text size */
    color: teal; /* Add this line to set text color to blue */
    letter-spacing: 0; /* Adjust if needed */
  }
</style> -->

<h1 align="center">
  <span style="color: teal; font-family: Consolas;">EQ-VAE</span>: Equivariance Regularized Latent Space for Improved Generative Image Modeling
</h1>




<div align="center">
  <a href="https://scholar.google.com/citations?user=a5vkWc8AAAAJ&hl=en" target="_blank">Theodoros&nbsp;Kouzelis</a><sup>1,3</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=B_dKcz4AAAAJ&hl=el" target="_blank">Ioannis&nbsp;Kakogeorgiou</a><sup>1</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.fr/citations?user=7atfg7EAAAAJ&hl=en" target="_blank">Spyros&nbsp;Gidaris</a><sup>2</sup> &ensp; <b>&middot;</b> &ensp;
  <a href="https://scholar.google.com/citations?user=xCPoT4EAAAAJ&hl=en" target="_blank">Nikos&nbsp;Komodakis</a><sup>1,4,5</sup>  
  <br>
  <sup>1</sup> Archimedes/Athena RC &emsp; <sup>2</sup> valeo.ai &emsp; <sup>3</sup> National Technical University of Athens &emsp; <br>
  <sup>4</sup> University of Crete &emsp; <sup>5</sup> IACM-Forth &emsp; <br>

<p></p>
<a href="https://eq-vae.github.io/"><img 
src="https://img.shields.io/badge/-Webpage-blue.svg?colorA=333&logo=html5" height=25em></a>
<a href="https://arxiv.org/abs/2502.09509"><img 
src="https://img.shields.io/badge/-Paper-blue.svg?colorA=333&logo=arxiv" height=25em></a>
<p></p>

![teaser.png](media/teaser.png)


</div>



<br>

<b>TL;DR</b>: We propose **EQ-VAE**, a straightforward regularization objective that promotes equivariance in the latent space of pretrained autoencoders under scaling and rotation. This leads to a more structured latent distribution, which accelerates generative model training and improves performance.


### 0. Quick Start with Hugging Face
If you just want to use EQ-VAE to speedup 🚀 the training on your diffusion model you can use our HuggingFace checkpoints 🤗.
We provide two models [eq-vae](https://huggingface.co/zelaki/eq-vae)
and [eq-vae-ema](https://huggingface.co/zelaki/eq-vae-ema). 

| Model   | Basemodel  | Dataset   | Epochs | rFID    | PSNR   | LPIPS  | SSIM   |
|---------|-------------|-----------|--------|--------|--------|--------|--------|
|  [eq-vae](https://huggingface.co/zelaki/eq-vae) | SD-VAE  | OpenImages | 5     | 0.82 | 25.95 | 0.141 | 0.72|
| [eq-vae-ema](https://huggingface.co/zelaki/eq-vae-ema)  | SD-VAE  | Imagenet | 44    | 0.55  | 26.15  | 0.133 | 0.72 |


```python
from diffusers import AutoencoderKL
eqvae = AutoencoderKL.from_pretrained("zelaki/eq-vae")
```

### 1. Environment setup

```bash
conda env create -f environment.yml
conda activate eqvae
```


### 2. Train EQ-VAE
We provide a training script to finetune [SD-VAE](https://ommer-lab.com/files/latent-diffusion/kl-f8.zip) with EQ-VAE regularization. For detailed guide go to [train_eqvae](./train_eqvae/).


### 3. Evaluate Reconstruction 
To evaluate the reconstruction of EQ-VAE, calculate rFID, LPIPS, SSIM and PSNR on a validation set (we use Imagenet Validation in our paper) with the following:
```bash
torchrun --nproc_per_node=8 eval.py \
  --data_path /path/to/imagenet/validation \
  --output_path results \
  --ckpt_path /path/to/your/ckpt
```

### 4. Train DiT with EQ-VAE
To train a DiT model with EQ-VAE on ImageNet:
  - First extract the latent representations:
  ```bash
  torchrun --nnodes=1 --nproc_per_node=8  train_gen/extract_features.py \
      --data-path /path/to/imagenet/train \
      --features-path /path/to/latents \
      --vae-ckpt /path/to/eqvae.ckpt \
      --vae-config configs/eqvae_config.yaml 
  ```
  - Then train DiT on the precomputed latents:
  ```bash
  accelerate launch --mixed_precision fp16 train_gen/train.py \
      --model DiT-XL/2 \
      --feature-path /path/to/latents \
      --results-dir results
  ```
  - Evaluate generation as follows:
  ```bash
  torchrun --nnodes=1 --nproc_per_node=8 sample_ddp.py \
      --model DiT-XL/2 \
      --num-fid-samples 50000 \
      --ckpt /path/to/dit.cpt \
      --sample-dir samples \
      --vae-ckpt /path/to/eqvae.ckpt \
      --vae-config configs/eqvae_config.yaml \
      --ddpm True \
      --cfg-scale 1.0
  ```

This script generates a folder of 50k samples as well as a .npz file and directly used with [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) to compute gFID.








### Acknowledgement

This code is mainly built upon [LDM](https://github.com/CompVis/latent-diffusion) and [fastDiT](https://github.com/chuanyangjin/fast-DiT). 
