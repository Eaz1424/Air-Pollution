# MSGAT: Multi-Scale Graph-Attention Transformer for Air Quality Forecasting

Forecast six pollutants from the UCI AirQuality dataset using a multi-scale Conv1D + feature (graph) attention + transformer encoder architecture. Includes training/plots, per-pollutant metrics, and paper drafts (Markdown + LaTeX).

## Project Layout

- `msgat_air_quality.py` — data prep, model, training, evaluation, plots.
- `AirQualityUCI.csv` — dataset (UCI Air Quality). Uses `;` delimiter, comma decimals.
- Generated outputs: `training_history.png`, `prediction_comparison.png`, `scatter_plot.png` (after running the script).

## Setup

```bash
# Recommended: new venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install tensorflow scikit-learn pandas matplotlib
```

## Run Training & Evaluation

From the project root:

```bash
python msgat_air_quality.py
```

What it does:

- Loads `AirQualityUCI.csv`, replaces `-200` with NaN, forward-fills, drops remaining NaNs.
- Uses features: CO(GT), NMHC(GT), NOx(GT), NO2(GT), PT08.S5(O3) (O3 proxy), C6H6(GT), T, RH.
- Window: 168 hours; predicts next-step values for the six pollutants.
- Model: multi-scale Conv1D (k=3,24) → feature self-attention (graph proxy) → transformer encoder → six dense heads.
- Regularization: dropout + L2; early stopping (patience 10, restore best weights).
- Outputs:
  - Console: scaled RMSE/MAE and per-pollutant RMSE/MAE in physical units.
  - Plots: `training_history.png`, `prediction_comparison.png` (benzene first 200 points), `scatter_plot.png`.

## Understanding Metrics

- Printed RMSE/MAE during training are in standardized space (z-scored).
- Per-pollutant metrics printed at the end are inverse-transformed to native units:
  - CO(GT): mg/m³
  - NMHC(GT): ppb
  - C6H6(GT): µg/m³
  - NOx(GT): ppb
  - NO2(GT): µg/m³
  - PT08.S5(O3): sensor A.U. (not direct concentration)


## Notes

- If your CSV has different column names, update `FEATURE_COLUMNS`/`POLLUTANT_HEADS` in `msgat_air_quality.py`.
- To shorten runs, lower `epochs` or batch size; to improve accuracy, train longer or tune filters/heads.
- GPU is recommended for speed but CPU works for the demo (longer runtime).

## Citation Placeholder

If you use this work, please cite the MSGAT paper draft once finalized.
