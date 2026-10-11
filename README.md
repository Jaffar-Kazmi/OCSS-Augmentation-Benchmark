# OCSS Augmentation Benchmark

Code for the paper "Comparative Analysis of Data Augmentation Strategies and Backbone Architectures for Class-Imbalanced Oral Lesion Classification" by Syed Jaffar Raza Kazmi, Riphah International University, Islamabad.

## What this repository does

The study compares ten training configurations for four-class oral lesion classification (Healthy, Benign, OPMD, OCA) on the Piyarathne smartphone oral cavity dataset. Eight configurations use ResNet50 with different augmentation or loss choices, and two use ConvNeXt-Tiny and ViT-B/16. All configurations share one five-fold, patient-grouped cross-validation split and are scored on pooled out-of-fold predictions. The repository also contains a sensitivity check that selects checkpoints on a separate inner validation split instead of the evaluated fold.

## Dataset

The images are not included in this repository. The dataset is described in N. S. Piyarathne, S. N. Liyanage and R. M. S. G. K. Rasnayaka, "A comprehensive dataset of annotated oral cavity images for diagnosis of oral cancer and oral potentially malignant disorders," Oral Oncology, vol. 156, p. 106946, 2024. Download it from the authors' release and follow their terms of use. Only image-level labels are used in this work, and the lesion polygons are not used.

## Repository layout

The `preprocessing/` folder holds the quality-check and image caching notebooks. The `experiments/` folder holds one notebook per configuration, each a copy of the shared baseline pipeline in which only the dataset and transform cell changes.

## Run order

Run the notebooks in the order below. Each configuration notebook reads the split file from step 2 and the cached images from step 1. The notebook file names in your copy of the repository may differ slightly from the descriptions here.

| Step | Notebook | Output |
|---|---|---|
| 1 | Preprocessing and quality control | Cleaned images, cached at 256x256 |
| 2 | Patient-grouped, class-stratified five-fold split | `piyarathne_split.csv` with a `fold` column (0 to 4) |
| 3 | Shared baseline pipeline, configuration 1 (no augmentation) | Results log row and out-of-fold predictions |
| 4 | Configurations 2 to 8 (ResNet50) | One results log row per configuration |
| 5 | Configurations 9 and 10 (ConvNeXt-Tiny, ViT-B/16) and the seed 123 rerun of 9 | One results log row each |
| 6 | Ensembles and error analysis | Ensemble scores and the persistent-error counts |
| 7 | Sensitivity check, configurations 1 and 4 with an inner validation split | Two results log rows |

## Configurations

| Cfg | Backbone | Transformation | Loss |
|---|---|---|---|
| 1 | ResNet50 | None | Weighted cross-entropy |
| 2 | ResNet50 | Geometric | Weighted cross-entropy |
| 3 | ResNet50 | Photometric | Weighted cross-entropy |
| 4 | ResNet50 | Geometric and photometric | Weighted cross-entropy |
| 5 | ResNet50 | MixUp (alpha 0.4) | Weighted cross-entropy |
| 6 | ResNet50 | CutMix (alpha 1.0) | Weighted cross-entropy |
| 7 | ResNet50 | RandAugment (N=2, M=9) | Weighted cross-entropy |
| 8 | ResNet50 | Geometric and photometric | Weighted focal loss (gamma 2) |
| 9 | ConvNeXt-Tiny | Geometric and photometric | Weighted cross-entropy |
| 10 | ViT-B/16 | Geometric and photometric | Weighted cross-entropy |

## Training setup

All models start from ImageNet-pretrained weights loaded through timm, with a new four-class head. Training uses AdamW with weight decay 1e-4, cosine annealing over at most 30 epochs, batch size 32, and early stopping after five epochs without improvement in validation macro-F1. The learning rate is 3e-4 for ResNet50, 1e-4 for ConvNeXt-Tiny and 5e-5 for ViT-B/16. The seed is 42, and the second ConvNeXt-Tiny run uses seed 123. Class weights are inversely proportional to class frequency in the training data.

## Main results

Pooled out-of-fold results with seed 42 and the original protocol, in which checkpoints are selected on the evaluated fold. These scores are optimistic by an amount that was not measured.

| Cfg | Strategy | Accuracy | Macro-F1 | OCA recall |
|---|---|---|---|---|
| 1 | ResNet50, no augmentation | 0.5830 | 0.4893 | 0.1783 |
| 2 | ResNet50, geometric | 0.5880 | 0.5195 | 0.2791 |
| 3 | ResNet50, photometric | 0.5633 | 0.5119 | 0.3488 |
| 4 | ResNet50, combined | 0.5850 | 0.5372 | 0.4031 |
| 5 | ResNet50, MixUp | 0.5840 | 0.5248 | 0.3488 |
| 6 | ResNet50, CutMix | 0.5947 | 0.5273 | 0.2946 |
| 7 | ResNet50, RandAugment | 0.6043 | 0.5337 | 0.3333 |
| 8 | ResNet50, combined with focal loss | 0.5807 | 0.5285 | 0.3566 |
| 9 | ConvNeXt-Tiny | 0.6147 | 0.5687 | 0.4419 |
| 10 | ViT-B/16 | 0.6113 | 0.5570 | 0.4109 |

## Sensitivity check

In this protocol a patient-grouped, class-stratified sixth of each training set is held out as an inner validation set, used only for early stopping and checkpoint selection. Each outer test fold is evaluated once after training. Single seed, no confidence intervals.

| Cfg | Accuracy | Macro-F1 | OCA recall | OCA F1 | OCA precision |
|---|---|---|---|---|---|
| 1, no augmentation | 0.5890 | 0.5008 | 0.2016 | 0.2626 | 0.3768 |
| 4, combined augmentation | 0.5413 | 0.4923 | 0.4574 | 0.3296 | 0.2576 |

The OCA recall gain from combined augmentation remains under this protocol, but the macro-F1 advantage does not.

## Software environment

The ResNet50 and ConvNeXt-Tiny runs used Kaggle with Python 3.12.13, PyTorch 2.10.0 (CUDA 12.8), torchvision 0.25.0, timm 1.0.26 and a Tesla T4 GPU. The ViT-B/16 run used Colab with Python 3.13.15, PyTorch 2.11.0 (CUDA 13.0), torchvision 0.26.0, timm 1.0.30 and a Tesla T4 GPU. The two sensitivity runs also used Kaggle with a Tesla T4 GPU. Deterministic cuDNN is enabled in all runs.

## Limitations

Results come from a single seed except for one ConvNeXt-Tiny rerun. The dataset has only 129 OCA images from 61 patients, so small differences between configurations are not reliable. The backbone comparison is confounded with the learning rate, ensembles were defined after the single-model results were known, and no external validation has been run. See the paper for the full list.

## Citation

```bibtex
@misc{kazmi2026augmentation,
  title  = {Comparative Analysis of Data Augmentation Strategies and Backbone Architectures for Class-Imbalanced Oral Lesion Classification},
  author = {Kazmi, Syed Jaffar Raza},
  year   = {2026},
  note   = {IEEE conference paper}
}
```

## License

Released under the MIT License. Add a `LICENSE` file with the standard MIT text before publishing.

## Contact

Syed Jaffar Raza Kazmi, 57375@students.riphah.edu.pk
