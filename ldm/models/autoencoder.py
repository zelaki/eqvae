import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution

from ldm.util import instantiate_from_config
import random
import json
import os
import copy
from diffusers import AutoencoderKL as DiffusersAutoencoderKL



def flip_or_rotate_image(inputs, flip):
    if flip == "h":
        inputs = torch.flip(inputs, [-1])

    elif flip == "v":
        inputs = torch.flip(inputs, [-2])

    elif flip == "vh":
        inputs = torch.flip(inputs, [-1,-2])

    elif flip == "90":
        inputs = torch.rot90(inputs, k=1, dims=[-1, -2]) 
    
    else:
        inputs = torch.rot90(inputs, k=3, dims=[-1, -2]) 

    return inputs


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 anisotropic=False,
                 uniform_sample_scale=True,
                 ckpt_path=None,
                 p_prior=0.5,
                 p_prior_s=0.25,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.uniform_sample_scale = uniform_sample_scale
        self.anisotropic = anisotropic
        self.p_prior=p_prior
        self.p_prior_s=p_prior_s
        assert ddconfig["double_z"]
        self.quant_conv = torch.nn.Conv2d(2*ddconfig["z_channels"], 2*embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, scale=1, angle=0):
        posterior = self.encode(input)
        z = posterior.sample()

        if scale != 1:            
            z = torch.nn.functional.interpolate(z, scale_factor=scale, mode='bilinear', align_corners=False)

        if angle != 0:
            z = torch.rot90(z, k=angle, dims=[-1, -2]) 

        dec = self.decode(z)
        return dec, posterior, z

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)

        # EQ-VAE regularization 
        if random.random() < self.p_prior:

            mode = "latent"
            if self.anisotropic:
                scale_x = random.choice([s / 32 for s in range(8,32)])
                scale_y = random.choice([s / 32 for s in range(8,32)])
                scale=(scale_x, scale_y)
            else:
                scale = random.choice([s / 32 for s in range(8,32)])

            # rotation angles 1 -> π/2, 2 -> π, 3 -> 3π/2
            angle = random.choice([1, 2, 3])
            reconstructions, posterior, z_after = self(inputs, scale=scale, angle=angle)

            # Scale ground truth images with the same scale
            inputs = torch.nn.functional.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
            
            # Rotate ground truth images with the same angle
            inputs = torch.rot90(inputs, k=angle, dims=[-1, -2]) 

        # prior preservation
        else:
            mode = "image"
            # this is prior preservation for low resolution images
            if random.random() < self.p_prior_s:

                scale = random.choice([s / 32 for s in range(8,32)])
                inputs = torch.nn.functional.interpolate(inputs, scale_factor=scale, mode='bilinear', align_corners=False)
                reconstructions, posterior, _ = self(inputs)

            # this is prior preservation for full resolution images
            else:
                scale=1
                reconstructions, posterior, _ = self(inputs)



        if optimizer_idx == 0:

            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")


            self.log(f"aeloss_scale-{scale}-{mode}", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:

            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")


            self.log(f"discloss_scale-{scale}-{mode}", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior, _ = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            if random.random() < 0.5:
                xrec, posterior, _ = self(x)
            else:
                xrec, posterior, _ = self(x, scale=0.5)

            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


