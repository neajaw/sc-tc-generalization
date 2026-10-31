# Assessing Generalization in Superconducting Critical Temperature Prediction Across Chemical Systems

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neajaw/sc-tc-generalization/blob/main/notebooks/tc_prediction.ipynb)

Machine learning models can predict the critical temperature (Tc) of known superconductors with high accuracy. This project asks a harder question: **do those models still work on chemical families they have never seen?**

## What this project does

1. Predicts Tc from 81 composition-derived features using:
   - **Classical ML:** Linear Regression, Random Forest, XGBoost
   - **Deep learning:** a PyTorch multilayer perceptron with dropout and early stopping
2. Evaluates with a **grouped split** by chemical formula, so the same compound never appears in both training and test sets.
3. Interprets the best model with **SHAP** to identify which physical properties drive Tc.
4. Tests **generalization across chemical systems** with leave-one-family-out evaluation over three families: cuprates, iron-based superconductors and others.

## Dataset

[UCI Superconductivity Data](https://archive.ics.uci.edu/dataset/464/superconductivty+data): 21,263 superconductors with 81 features and measured Tc. The notebook downloads it automatically.

- `train.csv`: 81 features plus `critical_temp`
- `unique_m.csv`: element counts, `critical_temp` and the chemical formula (`material`), in the same row order

## How to run

**Google Colab (recommended):** click the badge above, set **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

**Locally:**

```bash
pip install -r requirements.txt
jupyter notebook notebooks/tc_prediction.ipynb
```

Figures are saved to `figures/` and the metrics table to `results/results.csv`.

## Repository structure

```
sc-tc-generalization/
├── notebooks/
│   └── tc_prediction.ipynb   # full pipeline: data → ML/DL → SHAP → generalization
├── figures/                  # generated plots
├── results/                  # generated metrics (results.csv)
├── requirements.txt
└── README.md
```

## Results

| Model | Grouped random split RMSE (K) | Leave out Cuprate | Leave out Iron-based | Leave out Other |
|---|---|---|---|---|
| Linear Regression | | | | |
| Random Forest | | | | |
| XGBoost | | | | |
| MLP (deep learning) | | | | |

*Fill in from `results/results.csv` after running the notebook.*

## Reference

Hamidieh, K. (2018). A data-driven statistical model for predicting the critical temperature of a superconductor. *Computational Materials Science*, 154, 346–354.
