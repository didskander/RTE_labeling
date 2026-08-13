import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# =============================================================================
# PATHS
# =============================================================================
DATA_PATH = r"C:\Users\didsk\Desktop\Relay-protection\labeling_work\src\data\rte_events\DATA_S.npz"
LABELS_CSV = r"C:\Users\didsk\Desktop\Relay-protection\error_detection\data\processed\labels_first500_manual.csv"
OUTPUT_DIR = r"C:\Users\didsk\Desktop\Relay-protection\error_detection\models\cnn_baseline"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SETTINGS
# =============================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20      # fraction of trainval
BATCH_SIZE = 16
EPOCHS = 50

CONVERT_FROM_COUNTS = False
VOLTAGE_STEP = 18.310
CURRENT_STEP = 4.314

# optional downsampling for lighter training
DOWNSAMPLE_FACTOR = 4   # 21000 -> 5250 samples

# keep only labeled rows
VALID_LABELS = {"FAULT", "NORMAL", "UNCERTAIN"}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data_npz(path):
    with np.load(path, allow_pickle=False) as data:
        if "DATA_S" in data.files:
            arr = data["DATA_S"]
        elif "arr_0" in data.files:
            arr = data["arr_0"]
        else:
            arr = data[data.files[0]]

    arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 3 or arr.shape[1] != 6:
        raise ValueError(f"Expected shape (N, 6, T), got {arr.shape}")

    return arr


def convert_units(data):
    data = data.copy()
    if CONVERT_FROM_COUNTS:
        data[:, 0:3, :] *= VOLTAGE_STEP
        data[:, 3:6, :] *= CURRENT_STEP
    return data


def load_labels(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["label"].isin(VALID_LABELS)].copy()
    df = df.sort_values("event_id").reset_index(drop=True)
    return df


def downsample_signals(x, factor):
    if factor <= 1:
        return x
    return x[:, :, ::factor]


def normalize_per_event(x):
    """
    Robust per-event normalization:
    for each event and channel, subtract median and divide by IQR.
    """
    med = np.median(x, axis=2, keepdims=True)
    q75 = np.percentile(x, 75, axis=2, keepdims=True)
    q25 = np.percentile(x, 25, axis=2, keepdims=True)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0
    x_norm = (x - med) / iqr
    return x_norm.astype(np.float32)


def prepare_dataset(data_path, labels_csv):
    all_data = load_data_npz(data_path)
    all_data = convert_units(all_data)

    labels_df = load_labels(labels_csv)

    event_ids = labels_df["event_id"].values.astype(int)
    labels = labels_df["label"].values

    x = all_data[event_ids]                     # shape: (N, 6, T)
    x = downsample_signals(x, DOWNSAMPLE_FACTOR)
    x = normalize_per_event(x)

    # Keras Conv1D expects (N, T, C)
    x = np.transpose(x, (0, 2, 1))

    le = LabelEncoder()
    y = le.fit_transform(labels)

    return x, y, labels_df, le


# =============================================================================
# MODEL
# =============================================================================
def build_model(input_shape, n_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=9, padding="same", activation="relu")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=7, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.20)(x)

    outputs = layers.Dense(n_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# =============================================================================
# TRAINING
# =============================================================================
def main():
    x, y, labels_df, le = prepare_dataset(DATA_PATH, LABELS_CSV)

    print("X shape:", x.shape)
    print("y shape:", y.shape)
    print("Classes:", list(le.classes_))

    # train/test split
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # train/val split
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_trainval
    )

    model = build_model(input_shape=x_train.shape[1:], n_classes=len(le.classes_))
    model.summary()

    cb = [
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6
        ),
        callbacks.ModelCheckpoint(
            filepath=os.path.join(OUTPUT_DIR, "best_model.keras"),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test acc : {test_acc:.4f}")

    # Predictions
    y_prob = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    pred_df = pd.DataFrame({
        "true_label": le.inverse_transform(y_test),
        "pred_label": le.inverse_transform(y_pred),
        "pred_confidence": np.max(y_prob, axis=1)
    })
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # Save label encoder mapping
    class_map = {int(i): cls for i, cls in enumerate(le.classes_)}
    with open(os.path.join(OUTPUT_DIR, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=2)

    # Save training history
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv(os.path.join(OUTPUT_DIR, "training_history.csv"), index=False)

    print(f"Saved model and outputs to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()