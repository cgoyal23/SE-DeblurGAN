# SE-DeblurGAN

**Image deblurring with squeeze-and-excitation attention and a hybrid training objective.**

SE-DeblurGAN is a PyTorch notebook project that learns to restore sharp RGB images from paired blurred inputs. It combines a U-Net-style generator with squeeze-and-excitation (SE) attention, a conditional PatchGAN discriminator, and pixel, perceptual, frequency-domain, and adversarial losses.

The experiment uses the **CelebA subset of DBlur** and evaluates restoration quality with **PSNR** and **SSIM**.

[Explore the notebook](<image-deblurring-using-gan-a655e6%20(3).ipynb>)

## Highlights

- **SE attention on seven skip connections** to reweight encoder feature channels before decoding.
- **Conditional PatchGAN discriminator** that evaluates a blurred image together with its sharp or generated counterpart.
- **Hybrid generator loss** combining adversarial supervision, L1 reconstruction, VGG16 perceptual features, and FFT magnitude matching.
- **Training utilities** including paired horizontal flips, blurred-input color jitter, label smoothing, mixed precision, and cosine learning-rate scheduling.
- **Evaluation and visualization** with input-versus-output metrics, training curves, restored examples, and absolute difference maps.

## Model overview

```text
Blurred RGB image (256 × 256)
             │
       U-Net encoder
             │
          Bottleneck
             │
       U-Net decoder ← SE-weighted encoder skip connections
             │
 Concatenate with original input
             │
       Convolution + Tanh
             │
    Restored RGB image
```

The discriminator receives the blurred image concatenated with either the ground-truth sharp image or the generated image, forming a six-channel input.

### Training objective

```text
Generator loss = adversarial loss
               + 100 × L1 reconstruction loss
               +  10 × VGG16 perceptual loss
               +  10 × FFT magnitude loss
```

The adversarial term uses binary cross-entropy with logits. The perceptual term compares three groups of frozen ImageNet-pretrained VGG16 features with weights of 1, 1, and 2. The frequency term applies L1 loss to the magnitudes of the two-dimensional Fourier transforms.

## Recorded results

The committed notebook contains the following test-set outputs:

| Metric | Blurred input | Model output | Improvement |
| --- | ---: | ---: | ---: |
| PSNR ↑ | 19.53 dB | 26.33 dB | +6.80 dB |
| SSIM ↑ | 0.6036 | 0.7905 | +0.1869 |

The saved outputs report **50,000 training pairs** and **1,300 test pairs**. Metrics are computed after mapping images from `[-1, 1]` to `[0, 1]`.

**Reproducibility note:** these are saved notebook results, not a fresh benchmark. The saved training log shows a resumed run through epoch 10, while the current source sets `NUM_EPOCHS = 3` and contains no resume logic. The repository does not include the trained checkpoints, so running the current source does not reproduce that recorded run exactly. The ablation table is a template with unfilled values; it does not establish the contribution of individual components.

## Repository contents

| File | Description |
| --- | --- |
| [Deblurring notebook](<image-deblurring-using-gan-a655e6%20(3).ipynb>) | Dataset loading, model definitions, training, evaluation, and saved visualizations |
| `README.md` | Project overview and usage guidance |
| `*.pdf` | Additional CS771 coursework documents |

Datasets, trained weights, and a dependency lockfile are not included.

## Getting started

### 1. Set up the environment

The notebook was saved with **Python 3.11.13** in a **Kaggle environment with an NVIDIA Tesla T4 GPU**. Use a CUDA-capable PyTorch installation for the existing training cells, which explicitly use CUDA mixed precision.

For a local environment with a compatible PyTorch installation:

```bash
git clone https://github.com/cgoyal23/SE-DeblurGAN.git
cd SE-DeblurGAN

python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision torchmetrics numpy pandas matplotlib pillow tqdm jupyterlab
jupyter lab
```

These dependencies are inferred from the notebook imports and are not version-pinned. Ensure that PyTorch and torchvision are compatible with your system and CUDA installation. The VGG16 weights require internet access on first use unless they are already cached.

Alternatively, import the notebook into Kaggle, enable a GPU accelerator, and attach the dataset described below.

### 2. Prepare paired data

The notebook expects the CelebA portion of DBlur under the Kaggle input directory named `a-curated-list-of-image-deblurring-datasets`:

```text
DBlur/
└── CelebA/
    ├── train/
    │   ├── blur/
    │   └── sharp/
    └── test/
        ├── blur/
        └── sharp/
```

Each blurred image must have a sharp counterpart with **the same filename**. Keep these directories limited to image files because the loader opens every listed entry.

Update the following variables if your dataset is stored elsewhere:

```python
Train_Blur_Dir  = "/path/to/DBlur/CelebA/train/blur"
Train_Sharp_Dir = "/path/to/DBlur/CelebA/train/sharp"
Test_Blur_Dir   = "/path/to/DBlur/CelebA/test/blur"
Test_Sharp_Dir  = "/path/to/DBlur/CelebA/test/sharp"
```

Images are converted to RGB, resized to **256 × 256**, and normalized to `[-1, 1]`. Training randomly selects 50,000 pairs with seed 42; the remaining training images are unused. If you have fewer than 50,000 training pairs, adjust the `random_split` sizes before running.

### 3. Correct the notebook assertion

In the generator-definition cell, the shape assertion currently ends with a trailing comma, which causes a Python syntax error. Replace that line with:

```python
assert _g(_x).shape == (1, 3, 256, 256)
```

This correction is required before rerunning the notebook.

### 4. Train and evaluate

Open `image-deblurring-using-gan-a655e6 (3).ipynb` and run the cells in order after configuring the paths and correcting the assertion.

| Setting | Current source value |
| --- | --- |
| Image size | 256 × 256 |
| Batch size | 16 |
| Training epochs | 3 |
| Optimizer | Adam |
| Initial learning rate | 0.0002 |
| Adam betas | (0.5, 0.999) |
| Learning-rate schedule | Cosine annealing to 0.000001 |
| Real-label target | 0.9 |
| Random seed | 42 |

Reduce `BATCH_SIZE` if GPU memory is limited. For environments that have trouble with multiprocessing, reduce the data loaders' `num_workers` from 4 to 0. CPU use requires adapting the CUDA-specific mixed-precision calls.

The notebook saves the following files in its working directory:

| Output | Contents |
| --- | --- |
| `generator_epoch_<epoch>.pth` | Generator parameters for each completed epoch |
| `discriminator_epoch_<epoch>.pth` | Discriminator parameters for each completed epoch |
| `training_curves.png` | Loss, PSNR, and SSIM plots |
| `visual_results.png` | Blurred inputs, restored outputs, ground truth, and difference maps |

The evaluation cell sets `BEST_EPOCH = NUM_EPOCHS`, so it loads the **final epoch**, rather than selecting a checkpoint using validation metrics. Checkpoints contain model parameters only; optimizer, scheduler, and mixed-precision scaler states are not saved.

## Restore a single image

After running the notebook's imports and model-definition cells and obtaining a trained generator checkpoint, run:

```python
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
])

model = Generator().to(DEVICE)
model.load_state_dict(
    torch.load("generator_epoch_3.pth", map_location=DEVICE, weights_only=True)
)
model.eval()

image = Image.open("blurred_image.png").convert("RGB")
inputs = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    restored = (model(inputs).squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)

transforms.ToPILImage()(restored).save("restored_image.png")
```

Change the checkpoint filename to match your training run. The saved image is 256 × 256 because inference uses the same resizing as training.

## Current scope

This repository provides a notebook-based experiment on paired CelebA images. It does not include a standalone training application, pretrained weights, or completed ablation experiments. Performance on other image domains has not been established by the committed results.
