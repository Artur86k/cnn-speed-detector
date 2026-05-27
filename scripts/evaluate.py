"""
Visualise CNN speed predictions on the held-out validation session.

Loads the trained Keras model + per-channel normalisation stats saved by
train.py, runs predictions on every window in the val session, then plots:
  (1) predicted vs GPS-truth speed as a time series
  (2) scatter pred vs truth with the y=x reference line
  (3) residual histogram
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

OUT = Path(r'E:\Project\cnn_speed_detector\samples\preprocessed')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows', default=str(OUT / 'windows.npz'))
    ap.add_argument('--model',   default=str(OUT / 'speed_cnn.keras'))
    ap.add_argument('--hist',    default=str(OUT / 'train_history.npz'))
    ap.add_argument('--val-session', type=int, default=2)
    ap.add_argument('--save',    default=str(OUT / 'speed_cnn_eval.png'))
    args = ap.parse_args()

    # ----- load -------------------------------------------------------------
    d   = np.load(args.windows, allow_pickle=True)
    hist= np.load(args.hist,    allow_pickle=True)
    X   = d['X'].astype(np.float32)
    y   = d['y'].astype(np.float32)
    sid = d['session_id'].astype(np.int16)
    t   = d['t_label_s'].astype(np.float32)

    mean = hist['norm_mean'].reshape(1, 1, -1).astype(np.float32)
    std  = hist['norm_std'] .reshape(1, 1, -1).astype(np.float32)

    val = sid == args.val_session
    Xv, yv, tv = X[val], y[val], t[val]
    order = np.argsort(tv)
    Xv, yv, tv = Xv[order], yv[order], tv[order]
    Xv_n = (Xv - mean) / std
    print(f"Val session {args.val_session}: {len(yv)} windows  t={tv[0]:.1f}-{tv[-1]:.1f} s")

    model = keras.models.load_model(args.model)
    pred  = model.predict(Xv_n, verbose=0).flatten()
    err   = pred - yv
    mae   = float(np.mean(np.abs(err)))
    rmse  = float(np.sqrt(np.mean(err**2)))
    bias  = float(np.mean(err))
    print(f"  MAE={mae:.3f} m/s ({mae*3.6:.2f} km/h)  RMSE={rmse:.3f}  bias={bias:+.3f}")

    # ----- plot -------------------------------------------------------------
    fig = plt.figure(figsize=(14, 8), constrained_layout=True)
    gs  = fig.add_gridspec(2, 3)

    # (1) time series, full width on top row
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(tv, yv*3.6,    '-',  color='#1565c0', linewidth=1.4, label='GPS truth')
    ax1.plot(tv, pred*3.6, '--', color='#e53935', linewidth=1.1, label='CNN prediction',
             alpha=0.85)
    ax1.fill_between(tv, (pred-mae)*3.6, (pred+mae)*3.6,
                     color='#e53935', alpha=0.10, label=f'±MAE ({mae*3.6:.1f} km/h)')
    ax1.set_xlabel('Time in session (s)')
    ax1.set_ylabel('Speed (km/h)')
    ax1.set_title(f'Session {args.val_session} — predicted vs GPS truth')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # (2) scatter
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(yv*3.6, pred*3.6, s=4, alpha=0.4, color='#1565c0')
    lim = max((yv*3.6).max(), (pred*3.6).max()) * 1.05
    ax2.plot([0, lim], [0, lim], 'k--', linewidth=1, label='y = x')
    ax2.set_xlabel('GPS truth (km/h)')
    ax2.set_ylabel('CNN prediction (km/h)')
    ax2.set_title(f'Scatter  MAE={mae*3.6:.1f} km/h  bias={bias*3.6:+.1f}')
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # (3) residual histogram
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(err*3.6, bins=60, color='#1565c0', alpha=0.85, edgecolor='white')
    ax3.axvline(0,            color='k',       linestyle='-',  linewidth=1)
    ax3.axvline(bias*3.6,     color='#e53935', linestyle='--', linewidth=1.3,
                label=f'mean {bias*3.6:+.1f} km/h')
    ax3.set_xlabel('Residual pred−truth (km/h)')
    ax3.set_ylabel('Count')
    ax3.set_title('Residual distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # (4) MAE per speed bin
    ax4 = fig.add_subplot(gs[1, 2])
    bins   = np.array([0, 1, 5, 15, 25, 35])
    labels = ['0-1', '1-5', '5-15', '15-25', '25-35']
    bin_idx = np.digitize(yv, bins) - 1
    maes, counts = [], []
    for bi in range(len(labels)):
        m = bin_idx == bi
        maes  .append(np.mean(np.abs(err[m])) * 3.6 if m.any() else 0)
        counts.append(int(m.sum()))
    bars = ax4.bar(labels, maes, color='#1565c0', alpha=0.85)
    for b, c in zip(bars, counts):
        ax4.text(b.get_x() + b.get_width()/2, b.get_height(),
                 f'n={c}', ha='center', va='bottom', fontsize=8)
    ax4.set_xlabel('Truth speed (m/s)')
    ax4.set_ylabel('MAE (km/h)')
    ax4.set_title('Error by speed bin')
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle('CNN speed-detection results', fontsize=12, fontweight='bold')
    fig.savefig(args.save, dpi=130)
    print(f"Saved {args.save}")
    plt.show()


if __name__ == '__main__':
    main()
