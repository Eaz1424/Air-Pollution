"""MSGAT: Multi-Scale Graph-Attention Transformer for Air Quality Forecasting.

This demo script:
- Loads AirQualityUCI.csv (expects to be in the same folder).
- Cleans -200 placeholders, forward-fills missing values.
- Uses six pollutants (CO, NMHC, NOx, NO2, O3 proxy, C6H6) plus T and RH.
- Creates 168-hour sequences to predict the next-step pollutant vector.
- Builds a Keras model with multi-scale Conv1D + feature self-attention + transformer.
- Trains for a few epochs and saves plots:
  * training_history.png
  * prediction_comparison.png (benzene first 200 test points)
  * scatter_plot.png (benzene actual vs predicted)

Note: The O3 proxy uses the PT08.S5(O3) sensor channel in the UCI dataset.
"""

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Constants
SEQ_LENGTH = 168
TEST_RATIO = 0.15
VAL_RATIO = 0.15  # applied on the train split
REG = tf.keras.regularizers.l2(1e-4)
FEATURE_COLUMNS = [
    "CO(GT)",
    "NMHC(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S5(O3)",  # O3 proxy
    "C6H6(GT)",
    "T",
    "RH",
]
POLLUTANT_HEADS = [
    "CO(GT)",
    "NMHC(GT)",
    "NOx(GT)",
    "NO2(GT)",
    "PT08.S5(O3)",
    "C6H6(GT)",
]


def load_and_clean(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file at {path}")

    # UCI file uses ';' as delimiter; fall back to ',' if needed.
    try:
        df = pd.read_csv(path, sep=";", decimal=",", low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=",", decimal=",", low_memory=False)

    # Drop unnamed columns introduced by delimiter quirks.
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    df = df.replace(-200, np.nan).ffill().dropna()
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[FEATURE_COLUMNS]


def make_sequences(data: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    target_idx = [FEATURE_COLUMNS.index(c) for c in POLLUTANT_HEADS]
    for i in range(len(data) - seq_len - 1):
        window = data[i : i + seq_len]
        target = data[i + seq_len, target_idx]
        xs.append(window)
        ys.append(target)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def temporal_transformer_block(
    x: tf.Tensor,
    num_heads: int,
    key_dim: int,
    ff_dim: int,
    dropout: float,
    reg: tf.keras.regularizers.Regularizer,
) -> tf.Tensor:
    attn_output = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=dropout
    )(x, x)
    x = tf.keras.layers.LayerNormalization()(x + attn_output)
    ff = tf.keras.layers.Dense(ff_dim, activation="relu", kernel_regularizer=reg)(x)
    ff = tf.keras.layers.Dropout(dropout)(ff)
    ff = tf.keras.layers.Dense(x.shape[-1], kernel_regularizer=reg)(ff)
    x = tf.keras.layers.LayerNormalization()(x + ff)
    return x


def build_model(
    seq_len: int, num_features: int, num_heads: int = 4, key_dim: int = 16
) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(seq_len, num_features), name="inputs")

    # Multi-scale temporal convolutions
    conv_short = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=3,
        padding="same",
        activation="relu",
        kernel_regularizer=REG,
    )(inp)
    conv_long = tf.keras.layers.Conv1D(
        filters=64,
        kernel_size=24,
        padding="same",
        activation="relu",
        kernel_regularizer=REG,
    )(inp)
    conv_cat = tf.keras.layers.Concatenate(name="multi_scale_concat")(
        [conv_short, conv_long]
    )
    conv_cat = tf.keras.layers.Dropout(0.2)(conv_cat)

    # Feature (graph) attention: attend across features per time step
    perm = tf.keras.layers.Permute((2, 1))(conv_cat)  # (batch, features, time)
    feature_attn = tf.keras.layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, dropout=0.1
    )(perm, perm)
    feature_attn = tf.keras.layers.LayerNormalization()(perm + feature_attn)
    back = tf.keras.layers.Permute((2, 1))(feature_attn)  # (batch, time, features)
    back = tf.keras.layers.Dropout(0.2)(back)

    # Temporal transformer encoder
    trans = temporal_transformer_block(
        back,
        num_heads=num_heads,
        key_dim=key_dim,
        ff_dim=128,
        dropout=0.1,
        reg=REG,
    )
    trans = tf.keras.layers.Dropout(0.2)(trans)

    pooled = tf.keras.layers.GlobalAveragePooling1D()(trans)

    heads = []
    for name in POLLUTANT_HEADS:
        safe = (
            name.lower()
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        h = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=REG)(pooled)
        h = tf.keras.layers.Dropout(0.2)(h)
        out = tf.keras.layers.Dense(1, name=f"pred_{safe}", kernel_regularizer=REG)(h)
        heads.append(out)
    outputs = tf.keras.layers.Concatenate(name="pollutant_outputs")(heads)

    model = tf.keras.Model(inputs=inp, outputs=outputs, name="MSGAT")

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae", rmse]
    )
    return model


def main():
    data_path = os.path.join(os.path.dirname(__file__), "AirQualityUCI.csv")
    df = load_and_clean(data_path)

    values = df.values.astype(np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)

    X, y = make_sequences(scaled, SEQ_LENGTH)
    n = len(X)
    test_size = int(n * TEST_RATIO)
    val_size = int((n - test_size) * VAL_RATIO)

    X_train = X[: n - test_size - val_size]
    y_train = y[: n - test_size - val_size]
    X_val = X[n - test_size - val_size : n - test_size]
    y_val = y[n - test_size - val_size : n - test_size]
    X_test = X[n - test_size :]
    y_test = y[n - test_size :]

    model = build_model(SEQ_LENGTH, len(FEATURE_COLUMNS))
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=2,
    )

    test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test RMSE: {test_rmse:.3f}")

    # Predictions and inverse scaling for interpretability
    y_pred = model.predict(X_test, verbose=0)
    # Reconstruct scaled array for inverse transform: place predictions into pollutant slots, zeros elsewhere
    recon = np.zeros((len(y_pred), len(FEATURE_COLUMNS)), dtype=np.float32)
    for i, idx in enumerate([FEATURE_COLUMNS.index(c) for c in POLLUTANT_HEADS]):
        recon[:, idx] = y_pred[:, i]
    inv_preds = scaler.inverse_transform(recon)[
        :, [FEATURE_COLUMNS.index(c) for c in POLLUTANT_HEADS]
    ]

    recon_true = np.zeros_like(recon)
    for i, idx in enumerate([FEATURE_COLUMNS.index(c) for c in POLLUTANT_HEADS]):
        recon_true[:, idx] = y_test[:, i]
    inv_true = scaler.inverse_transform(recon_true)[
        :, [FEATURE_COLUMNS.index(c) for c in POLLUTANT_HEADS]
    ]

    # Benzene (C6H6) index
    benz_idx = POLLUTANT_HEADS.index("C6H6(GT)")
    benz_true = inv_true[:, benz_idx]
    benz_pred = inv_preds[:, benz_idx]

    # Per-pollutant metrics in physical units
    print("Per-pollutant metrics (physical units):")
    for i, name in enumerate(POLLUTANT_HEADS):
        rmse_phys = mean_squared_error(inv_true[:, i], inv_preds[:, i], squared=False)
        mae_phys = mean_absolute_error(inv_true[:, i], inv_preds[:, i])
        print(f"  {name}: RMSE={rmse_phys:.3f}, MAE={mae_phys:.3f}")

    # Plot training history
    plt.figure(figsize=(6, 4))
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()

    # Prediction comparison (first 200 points)
    limit = min(200, len(benz_true))
    plt.figure(figsize=(8, 4))
    plt.plot(range(limit), benz_true[:limit], label="Actual")
    plt.plot(range(limit), benz_pred[:limit], label="Predicted")
    plt.xlabel("Time (hours)")
    plt.ylabel("Benzene (C6H6)")
    plt.title("Benzene Prediction: Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig("prediction_comparison.png", dpi=150)
    plt.close()

    # Scatter plot
    plt.figure(figsize=(5, 5))
    plt.scatter(benz_true, benz_pred, alpha=0.5, s=12)
    lims = [
        min(benz_true.min(), benz_pred.min()),
        max(benz_true.max(), benz_pred.max()),
    ]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Benzene Actual vs Predicted")
    plt.tight_layout()
    plt.savefig("scatter_plot.png", dpi=150)
    plt.close()

    print(
        "Plots saved: training_history.png, prediction_comparison.png, scatter_plot.png"
    )


if __name__ == "__main__":
    main()
