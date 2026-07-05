# U-Net-1D vs U-Net-2D — power-system disturbance detection

Self-contained code **and data** to reproduce the comparison between a
raw-waveform **U-Net-1D** and a spectrogram **U-Net-2D** for detecting,
localizing, and typing faults / power-quality disturbances — in **both** a
**three-phase** and a **single-phase** setting.

Both networks share identical channel widths `(8,16,32,64)`, so any performance
gap is due to the input **representation** (raw waveform vs. log-STFT
spectrogram), not model capacity. The U-Nets are implemented in **pure NumPy**
(`unet_numpy.py`) — no deep-learning framework is required, CPU only.

> Full, step-by-step run instructions are in **`INSTRUCTIONS.txt`**.

## Quick start

```bash
pip install -r requirements.txt

# three-phase (dataset in three_phase/data/)
cd three_phase && python3 run_unet_comparison.py

# single-phase (dataset in single_phase/data_det/)
cd single_phase && python3 run_detection.py
```

## Layout

| Path | What |
|------|------|
| `three_phase/run_unet_comparison.py` | Main 3-phase script: U-Net-1D vs U-Net-2D on the augmented IBR dataset (train/val on backgrounds A+B, test on unseen C). |
| `three_phase/data/` | Eight `.npz` dataset parts (6-channel V&I, 200 events). |
| `single_phase/run_detection.py` | Main single-phase script: U-Net-1D vs U-Net-2D with a 0/25/50/75 % STFT overlap sweep (train on days 1–5, test on unseen day 6). |
| `single_phase/data_det/` | `day1.npz … day6.npz`. |
| `*/unet_numpy.py` | Pure-numpy U-Net (1-D and 2-D) with a fault-type head. |

## Outputs

- **Three-phase:** metrics table in the console, `three_phase/plots/summary.json`,
  and figures (`training_curves`, `test_event_traces`, `pr_curves`,
  `ap_vs_iou`, `summary_bars`).
- **Single-phase:** overlap-sweep table, `single_phase/results_detection.json`,
  and figures (`det_examples`, `det_traces`, `det_pr_curves`, `det_ap_vs_iou`,
  `det_type_confusion`).

All seeds are fixed for reproducibility. Everything runs on CPU.
