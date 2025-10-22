# Amazon ML Challenge 2025


## Abstract 
We build a multimodal model that predicts product price from catalog text, optional product images, and engineered metadata. Our pipeline combines transformer-based text embeddings, ResNet-based image features, and tabular models (LightGBM + MLP ensemble) to achieve robust regression performance.

## Problem Statement
Given product-level inputs (catalog content, images, and available metadata), predict the selling price for each product in the evaluation set. Our objective in the challenge is to minimize the evaluation metric (e.g., RMSE/MAE) specified by the organizers on the holdout test set.

## Data
- Train file: CSV with columns including `sample_id`, `catalog_content`, `price`, and optional metadata.
- Test file: CSV with columns including `sample_id` and `catalog_content`.

(If you used external pretraining or auxiliary datasets, list them here and confirm they are allowed by the challenge rules.)

## Our Approach
High-level components:
1. Preprocessing & Feature Engineering
   - Cleaned `catalog_content` with light normalization (lowercase, whitespace, basic punctuation removal).
   - Extracted numeric pack/quantity features with regex (e.g., "100g", "x12").
   - Created content length and word count features.
   - Log-transformed the `price` target using log1p to stabilize variance. Capped the top 0.1% of log-prices to reduce outlier influence.

2. Text Representation
   - Used `sentence-transformers` (`all-mpnet-base-v2`) to generate 384-D sentence embeddings for `catalog_content`.



4. Model(s)
   - LightGBM regressor trained on concatenated engineered features + text embeddings. Objective: MAE (regression_l1) with RMSE monitoring for early stopping.
   - A Keras MLP for the same multimodal input (dense blocks with batch normalization and dropouts).
   - Final predictions: weighted ensemble of LightGBM and MLP outputs on log-target, inverse-transformed with expm1.




## Results (example)
- Local validation (10% holdout): RMSE = 0.543, MAE = 0.421 on log(1+price) scale.
- Public/private leaderboard (placeholder): Public LB = 0.62 (if available after competition).

Replace these numbers with your real validation and leaderboard scores.



## Hardware & Runtime
- Development: Colab GPU (Tesla T4 / P100) and local CPU for light testing.
- Typical training time: depends on dataset size and whether embeddings and image features are computed locally. Text embedding generation on GPU is recommended.

## Reproducibility & Notes
- Set random seeds where applicable (numpy, tensorflow, lightgbm) to improve reproducibility.
- Save preprocessing pipeline and model artifacts when finalizing your submission.
- Ensure the final `test_out.csv` follows the challenge format: two columns `sample_id,price`, with the `price` column formatted as a plain numeric value (rounded to 2 decimals if required).

## What we tried but did not include (ablation)
- Heavier text cleaning (didn't show consistent gains).
- Advanced image augmentations for ResNet features — complexity/compute tradeoff was not worth it for our runs.

## Future work
- Better target transformation (heteroscedastic models), feature interactions, and stacking multiple tree-based models.
- Calibrated quantile regression for more robust leaderboard performance.

## Acknowledgements
- Thanks to the Amazon AI & ML Challenge 2025 organizers for the host dataset and evaluation infrastructure.

---
