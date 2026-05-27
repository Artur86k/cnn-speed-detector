"""
End-to-end smoke test: train a small 1D-CNN to regress vehicle speed from
the (T=400, C=6) windows produced by dataset.py.

Goals:
  - Validate the preprocess -> window -> train pipeline end-to-end on the
    3-session corpus in samples/.
  - Get a baseline MAE so we know whether scaling to the 36-session corpus
    is worth doing.

Split:
  - Train: sessions 0 (74-min, noisy GPS) + 1 (3-min)         -> 1864 windows
  - Val  : session 2 (14-min, clean GPS  -- held out)          -> 1616 windows

Outputs to samples/preprocessed/:
  - speed_cnn.keras           (Keras model)
  - speed_cnn_smoketest.tflite
  - train_history.npz         (loss/MAE curves for later inspection)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Quiet TF logging before import
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DEFAULT_NPZ = r'E:\Project\cnn_speed_detector\samples\preprocessed\windows.npz'
OUT_DIR     = r'E:\Project\cnn_speed_detector\samples\preprocessed'


def build_cnn(T: int, C: int) -> keras.Model:
    """Small 1D-CNN: ~30k params, fits on phone, suited to 2 s @ 200 Hz."""
    inp = keras.Input(shape=(T, C), name='imu_window')
    x = layers.Conv1D(32, 7, strides=2, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 5, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv1D(64, 3, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu', name='embedding')(x)
    out = layers.Dense(1, name='speed_ms')(x)
    return keras.Model(inp, out, name='speed_cnn')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', default=DEFAULT_NPZ)
    ap.add_argument('--out',     default=OUT_DIR)
    ap.add_argument('--epochs',  type=int, default=40)
    ap.add_argument('--batch',   type=int, default=64)
    ap.add_argument('--val-session', type=int, default=2,
                    help='session_id held out for validation (default: 2 = cleanest)')
    args = ap.parse_args()

    # ----- Load -----------------------------------------------------------------
    d = np.load(args.windows, allow_pickle=True)
    X    = d['X'].astype(np.float32)              # (N, T, C)
    y    = d['y'].astype(np.float32)              # (N,)
    sid  = d['session_id'].astype(np.int16)
    chs  = list(d['channels'])
    print(f"Windows : N={X.shape[0]}  T={X.shape[1]}  C={X.shape[2]}  channels={chs}")
    print(f"Sessions: {sorted(set(int(s) for s in sid))}")

    val_mask   = sid == args.val_session
    train_mask = ~val_mask
    print(f"Train   : {int(train_mask.sum())} windows  (sessions {sorted(set(int(s) for s in sid[train_mask]))})")
    print(f"Val     : {int(val_mask.sum())} windows  (session {args.val_session})")
    if int(train_mask.sum()) < 100 or int(val_mask.sum()) < 100:
        print("  WARNING: small split — smoke test only, metrics will be noisy")

    Xtr, ytr = X[train_mask], y[train_mask]
    Xv,  yv  = X[val_mask],   y[val_mask]

    # ----- Per-channel normalisation (fit on train, apply to both) ---------------
    mean = Xtr.mean(axis=(0, 1), keepdims=True)
    std  = Xtr.std(axis=(0, 1), keepdims=True) + 1e-6
    Xtr  = (Xtr - mean) / std
    Xv   = (Xv  - mean) / std
    print(f"Norm    : mean={mean.flatten().round(3).tolist()}  std={std.flatten().round(3).tolist()}")

    # ----- Build + train --------------------------------------------------------
    T, C  = X.shape[1], X.shape[2]
    model = build_cnn(T, C)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss=keras.losses.Huber(delta=1.0),       # robust to noisy GPS labels
        metrics=['mae'],
    )
    model.summary(print_fn=print)

    cb = [
        keras.callbacks.EarlyStopping(monitor='val_mae', patience=8,
                                      restore_best_weights=True, mode='min'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=4, min_lr=1e-5),
    ]
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xv, yv),
        epochs=args.epochs,
        batch_size=args.batch,
        callbacks=cb,
        verbose=2,
        shuffle=True,
    )

    # ----- Evaluate -------------------------------------------------------------
    val_mae_ms = model.evaluate(Xv, yv, verbose=0)[1]
    pred = model.predict(Xv, verbose=0).flatten()
    err  = pred - yv
    print(f"\n=== Validation summary ===")
    print(f"  MAE : {val_mae_ms:.3f} m/s   = {val_mae_ms*3.6:.2f} km/h")
    print(f"  RMSE: {np.sqrt(np.mean(err**2)):.3f} m/s")
    print(f"  bias: {err.mean():+.3f} m/s")
    # error binned by speed
    bins = [(0, 1), (1, 5), (5, 15), (15, 25), (25, 50)]
    print(f"  by-speed MAE:")
    for lo, hi in bins:
        m = (yv >= lo) & (yv < hi)
        if m.sum() == 0: continue
        print(f"    {lo:2d}-{hi:2d} m/s  ({3.6*lo:.0f}-{3.6*hi:.0f} km/h):  "
              f"n={int(m.sum()):5d}  MAE={np.mean(np.abs(err[m])):.3f} m/s")

    # ----- Save -----------------------------------------------------------------
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keras_path  = out / 'speed_cnn.keras'
    tflite_path = out / 'speed_cnn_smoketest.tflite'
    hist_path   = out / 'train_history.npz'

    model.save(keras_path)
    print(f"\nSaved {keras_path}")

    # Save normalization stats alongside model history
    np.savez_compressed(
        hist_path,
        loss=np.array(hist.history['loss']),
        val_loss=np.array(hist.history['val_loss']),
        mae=np.array(hist.history['mae']),
        val_mae=np.array(hist.history['val_mae']),
        norm_mean=mean.squeeze(),
        norm_std=std.squeeze(),
        channels=np.array(chs),
    )
    print(f"Saved {hist_path}")

    # TFLite — float32, no dynamic-range quantization.
    # Optimize.DEFAULT made TF 2.21 emit FULLY_CONNECTED v12, which the app's
    # TFLite runtime (2.16.1) does not understand and refuses to load.
    # Plain float32 uses FC v9 which every 2.x runtime supports.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    tfl = converter.convert()
    tflite_path.write_bytes(tfl)
    print(f"Saved {tflite_path}  ({len(tfl)/1024:.1f} KB)")


if __name__ == '__main__':
    main()
