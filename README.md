# Scenario 1 - Landcover Classification (Delhi NCR)

This repo contains code and scripts for the Earth-Observation selection task (IIT Gandhinagar). The pipeline performs spatial gridding, constructs labels from ESA WorldCover, and trains a ResNet18 classifier on Sentinel-2 RGB patches.

**Data**
- *Not included in this repo.* Place the dataset under `data/` locally or use Google Drive. Expected files:
  - `data/delhi_ncr_region.geojson`
  - `data/delhi_airshed.geojson`
  - `data/worldcover_bbox_delhi_ncr_2021.tif`
  - `data/rgb/*.png` (128×128 Sentinel-2 patches named `lat_lon.png`)


## Quickstart (Google Colab recommended)
1. Mount Google Drive and place dataset in `MyDrive/earth-observation-iitgn-2025/data/`.
2. Install requirements: `pip install -r requirements.txt`.
3. Run the Colab notebook `notebooks/scenario1_pipeline.ipynb` step-by-step.

## Scripts
- `scripts/train_model.py` — train ResNet18 and save best weights.
- `scripts/evaluate.py` — evaluate saved model and produce confusion matrix, metrics.
- `scripts/utils.py` — helpers: patch extraction, mapping, dataset utils.

## Outputs
- `outputs/landcover_labeled_dataset.csv` — labeled dataset used for training.
- `outputs/train_split.csv`, `outputs/test_split.csv` — splits.
- `outputs/per_class_metrics.json`, `outputs/confusion_matrix_norm.png`.


