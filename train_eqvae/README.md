
### 1. Environment setup
```bash
conda env create -f environment.yaml
conda activate eqvae_train
pip install packaging==21.3
pip install 'torchmetrics<0.8'
pip install transformers==4.10.2
pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install Pillow==9.5.0
```


### 2. Download SD-VAE
To download the SD-VAE from the official LDM repository run:


```bash
bash download_sdvae.sh
```



### 3. Dataset

#### Dataset download

Currently, we provide experiments for [OpenImages](https://storage.googleapis.com/openimages/web/index.html). After downloading modify paths of train_dir, val_dir, dataset_name in the [cofig file](configs/eqvae_config.yaml)









### 4. Training

To run EQ-VAE regularization on 8 GPUs:

```bash
python main.py \
    --base configs/eqvae_config.yaml \
    -t \
    --gpus 0,1,2,3,4,5,6,7 \
    --resume pretrained_models/model.ckpt \
    --logdir logs/eq-vae   
```


Then this script will automatically create the folder in `logs/eq-vae` to save logs and checkpoints.
The provided arguments in `configs/eqvae_config.yaml` are the ones used in our paper.  You can adjust the following options for your experiments:

- `anisotropic`: If `True` will do anisotropic scaling 
- `uniform_sample_scale`: If `True` will sample scale factors uniformly from `[0.25, 1)` if set to `False` will randomly choose from scales from `{0.25, 0.5, 0.75}`.
- `p_prior`: Probability to do prior preservation instead of equivariance regularization
- `p_prior_s`: Probability to do prior presevation on lower resolutions instead of equivariance regularization
  