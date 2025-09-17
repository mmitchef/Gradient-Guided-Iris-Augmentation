# Gradient-Guided Exploration of Generative Model’s Latent Space for Controlled Iris Image Augmentations

This repository provides code for controlled augmentation of human iris images by performing gradient-guided exploration in the latent space of a pre-trained generative model.

## Installation
```
conda env create -f environment.yml
conda activate GGIA
```
## Usage

* Identify the attribute you want to modify (e.g., sharpness, pupil/iris size, pupil-to_iris_ratio).

* If you plan to use identity loss, set the --use_identity flag to True and specify the target value for the chosen attribute.

Example command:
```
python gradient_guided_iris_augmentation.py \
  --network=Network.pkl \
  --outdir=./output/ \
  --seed=<random_seed> \
  --sharp_inc=True \
  --use_identity=True \
  --target_val=5.0
```
--network : Path to the pre-trained network file.

--outdir : Output directory for augmented images.

--seed : Random seed for reproducibility.

--sharp_inc : Treu/False (select one of these iris_inc, iris_dec, pupil_inc, puil_dec, sharp_inc, sharp_dec, pir_inc, pir_dec)

--use_identity : Enable identity loss during augmentation.

--target_val : Target value for the selected attribute.

