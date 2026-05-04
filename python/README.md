# SNN Model Parameters & Simulation Summary

This directory contains the Python implementation of the **MultiStage-DeepSNN**, including the PyTorch reference model and a hardware-accurate simulation using Numba.

## 1. Network Architecture (G11)
The model is a three-stage Spiking Convolutional Neural Network followed by a Fully Connected classifier.

| Layer | Type | Hyperparameters | Output Shape (Spatial) |
| :--- | :--- | :--- | :--- |
| **Input** | Video | 16 frames, 256x256 Grayscale | (16, 256, 256, 1) |
| **Stage 1** | Conv + BN + Pool | 32 filters, 5x5 Kernel, Stride 1, Pad 2 | 128x128x32 |
| **LIF 1** | Spiking Neuron | $\beta \approx 0.9$, Threshold extracted from PT | 128x128x32 |
| **Stage 2** | Conv + BN + Pool | 64 filters, 3x3 Kernel, Stride 1, Pad 1 | 64x64x64 |
| **LIF 2** | Spiking Neuron | $\beta \approx 0.9$, Threshold extracted from PT | 64x64x64 |
| **Stage 3** | Conv + BN + Pool | 128 filters, 3x3 Kernel, Stride 1, Pad 1 | 32x32x128 |
| **LIF 3** | Spiking Neuron | $\beta \approx 0.9$, Threshold extracted from PT | 32x32x128 |
| **GAP** | Global Avg Pool | Average across spatial dimensions | 128 |
| **FC1** | Linear + ReLU | 128 $\rightarrow$ 256 | 256 |
| **FC2** | Linear | 256 $\rightarrow$ 4 (Classes) | 4 |

## 2. Hardware Simulation Parameters
The `CustomQuantizedG11` class simulates FPGA hardware behavior using the following fixed-point constraints:

| Parameter | Default Value | Description |
| :--- | :--- | :--- |
| `DATA_WIDTH` | 16-bit / 18-bit | Total bits used for signed fixed-point (Q format) |
| `FRAC_BITS` | 10 bits | Precision for fractional components |
| `BETA` ($\beta$) | 0.9 (PT) / 0.5 (RTL) | Leakage factor. Simulation uses PT value; RTL uses shift. |
| `THRESHOLD` | Extracted | Firing threshold determined during training |
| `DROPOUT` | 0.4 | Probability for dropout during training (disabled in eval) |
| `TEMPORAL_STEPS`| 16 | Number of frames processed in the spiking loop |

## 3. Quantization Range
Based on a Q7.10 (18-bit) or similar signed format:
- **Scale Factor:** $2^{10} = 1024$
- **Min Value:** $-2^{(TotalBits-1)} / Scale$
- **Max Value:** $(2^{(TotalBits-1)} - 1) / Scale$

## 4. Class Labels
The network classifies driving incidents into four distinct categories:

| Label | Class Name | Description |
| :--- | :--- | :--- |
| **0** | `negative_samples` | No incident detected |
| **1** | `drifting_or_skidding` | Risky maneuver, no impact |
| **2** | `other_crash` | Crash detected, ego vehicle not involved |
| **3** | `collision` | Crash involving the ego vehicle |

## 5. Execution Modes in `main.py`
1. **`ordi()`**: Performs a deep comparison between the PyTorch float model and the Numba-based hardware wrapper (quantization disabled) to verify logic equivalence.
2. **`final()`**: Iterates through various bit-widths to find the optimal quantization setting that maintains accuracy while minimizing hardware footprint.
3. **`final_big_test()`**: Runs a full evaluation on a 500-sample subset using recommended quantization (e.g., 19-bit or 20-bit total).

---
*Note: The Numba-accelerated functions (`myConv2D_numba`, etc.) implement channel-last data ordering to match standard hardware streaming patterns.*