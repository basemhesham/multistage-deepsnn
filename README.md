# MultiStage-DeepSNN: Hardware Architecture Documentation

---

## Table of Contents

1. [SNN Design Overview](#1-snn-design-overview)
2. [Target FPGA — Xilinx Virtex UltraScale+ XCVU11P](#2-target-fpga--xilinx-virtex-ultrascale-xcvu11p)
3. [Backtracking: Input Window Derivation](#3-backtracking-input-window-derivation)
4. [Stage 1: 5×5 Convolution Architecture](#4-stage-1-5x5-convolution-architecture)
5. [Stage 2: 3×3 Convolution — 32 Input Channels](#5-stage-2-3x3-convolution--32-input-channels)
6. [Stage 3: 3×3 Convolution — 64 Input Channels](#6-stage-3-3x3-convolution--64-input-channels)
7. [Memory Control & Frame Mapping](#7-memory-control--frame-mapping)
8. [Classifier Head: GAP, FC1, FC2](#8-classifier-head-gap-fc1-fc2)
9. [Full Resource Summary](#9-full-resource-summary)

---

## 1. SNN Design Overview

### Project Goal

**MultiStage-DeepSNN** is an FPGA hardware accelerator that implements a fully trained **Spiking Neural Network (SNN)** for real-time classification of driving incidents from dashcam video. The network identifies four event categories from a short clip:

| Label | Class | Meaning |
|-------|-------|---------|
| 0 | `negative_samples` | No incident detected |
| 1 | `drifting_or_skidding` | Risky manoeuvre, no impact |
| 2 | `other_crash` | Crash detected — ego vehicle not involved |
| 3 | `collision` | Crash — ego vehicle directly involved |

### Network Architecture

The network is a **three-stage spiking convolutional network** operating over **T = 16 temporal frames**. Each frame is a **256×256 grayscale image**. The network structure is:

```
Input (B, T, 256, 256, 1)
        │
        ▼  Permute to (B, T, C, H, W)
  ┌─────────────────────────────────────────────────┐
  │           Temporal Loop  t = 0 → T−1            │
  │                                                 │
  │   Stage 1 ─── Conv 5×5 (32 filters)             │
  │               BN + MaxPool 2×2                  │
  │               LIF 1  → spikes                   │
  │                   │                             │
  │   Stage 2 ─── Conv 3×3 (64 filters)             │
  │               BN + MaxPool 2×2                  │
  │               LIF 2  → spikes                   │
  │                   │                             │
  │   Stage 3 ─── Conv 3×3 (128 filters)            │
  │               BN + MaxPool 2×2                  │
  │               LIF 3  → spikes                   │
  └─────────────────────────────────────────────────┘
        │  Record spikes over all T frames
        ▼
  Global Average Pool (AdaptiveAvgPool 1×1)
        │
  FC1: 128 → 256 + ReLU + Dropout
        │
  FC2: 256 → 4
        │
  Class Logits (0–3)
```

<div align="center">
<img width="2000" height="564" alt="Picture1" src="https://github.com/user-attachments/assets/9d2d1e2a-42d9-4687-b119-73937a786371" />
</div>

Each of the three convolutional blocks follows the same pattern:

**Conv → BatchNorm → MaxPool 2×2 → LIF neuron**

The **Leaky Integrate-and-Fire (LIF)** neurons accumulate membrane potential across all T timesteps. At each step the membrane leaks by factor β (β = 0.9 during PyTorch training; β = 0.5 in RTL hardware using an arithmetic right-shift). A spike fires when the membrane potential reaches or exceeds the threshold.

### Stage Specifications

| Stage | Conv Kernel | Input Channels | Output Channels | Input Spatial | After Conv | After MaxPool |
|-------|------------|---------------|----------------|---------------|------------|---------------|
| 1 | 5×5 | 1 | 32 | 256×256 | 20×20 | 10×10 |
| 2 | 3×3 | 32 | 64 | 10×10 | 8×8 | 4×4 |
| 3 | 3×3 | 64 | 128 | 4×4 | 2×2 | 1×1 |

> Note: The spatial dimensions shown above correspond to one 24×24 crop processed per the backtracking approach (Section 3). The hardware does not process the full 256×256 feature maps at once.

### Design Philosophy

Rather than generating full spatial feature maps for every stage and buffering them between stages — which would introduce massive memory requirements and pipeline stalls — this design uses a **z-domain (depth-first) processing** strategy derived from backtracking analysis. The hardware generates only the data that each downstream stage immediately needs, allowing all three stages to operate concurrently on different parts of the same input. This approach:

- Eliminates the need to fully compute and store each stage's complete output before the next stage begins.
- Enables deep **pipeline parallelism** by exploiting pre-known filter weights (stored as constants) to process in the depth (z) dimension rather than waiting for the full spatial (x) dimension to complete.
- Minimizes on-chip memory usage for intermediate feature maps.

---

## 2. Target FPGA — Xilinx Virtex UltraScale+ XCVU11P

The design targets the **Xilinx Virtex UltraScale+ XCVU11P** in the `flga2577-3-e` package. This device belongs to the highest-capacity family of UltraScale+ FPGAs and provides the resources necessary to instantiate hundreds of parallel convolution units and their associated accumulation logic.

### Device Resources

| Resource Type | XCVU11P (VU11P) |
| :--- | :--- |
| **System Logic Cells** | 2,835,000 |
| **CLB Flip-Flops** | 2,592,000 |
| **CLB LUTs** | 1,296,000 |
| **Max. Distributed RAM (Mb)** | 36.2 |
| **Block RAM Blocks** | 2,016 |
| **Block RAM (Mb)** | 70.9 |
| **UltraRAM Blocks** | 960 |
| **UltraRAM (Mb)** | 270.0 |
| **DSP Slices** | 9,216 |

### DSP48E2 slice

The DSP48E2 slice is the central resource in this design. The accelerator maps virtually all multiply-accumulate and addition operations onto DSP48E2 primitives using their native capabilities:

- **Multiplier port:** 27×18-bit signed multiply inside one DSP slice.
- **Pre-adder port (D):** Allows a third input to be added before the multiplier, enabling a true 3-input adder in a single DSP and single pipeline stage.
- **Cascade bus:** DSP slices can chain their outputs directly to adjacent slices without routing fabric, forming efficient **cascaded MAC chains** for convolution accumulation.

<div align="center">
    <img width="689" height="443" alt="image" src="https://github.com/user-attachments/assets/b76843a0-f010-40c9-b490-559f95578cb0" />
</div>

### Operating Target

| Parameter | Value |
|-----------|-------|
| Target clock frequency | **40 MHz** |
| Fixed-point format | **18-bit signed, Q7.10** |
| Scale factor | 2¹⁰ = 1024 |
| Representable range | −128.000 to +127.999 |

---

## 3. Backtracking: Input Window Derivation

### Concept

Standard neural network inference processes the entire input image through each stage in sequence, storing the full feature map of each stage before advancing to the next. This is inefficient in hardware because it requires large memories to hold intermediate results and forces pipeline stages to stall waiting for predecessors.

The **backtracking approach** solves this by working from the desired output backwards through every layer to determine the smallest input window that produces exactly one output element at the final stage. The hardware then tiles this minimum window across the image, enabling all three stages to work simultaneously in a streaming fashion.

<div align="center">
<img width="511" height="362" alt="Backtracking" src="https://github.com/user-attachments/assets/6da5f8c4-45ac-4046-8464-998e2c8c85dc" />
</div>

### Stage 3: Final Output Dependency

To produce **1 LIF output element** from Stage 3:

- The **LIF and BN/MaxPool block** requires a **2×2 spatial window** as input (MaxPool 2×2 with stride 2 reduces 2×2 to 1×1, producing the single output element).
- The **Stage 3 Convolution** (3×3 kernel, stride 1, no padding) must therefore generate these **4 elements** arranged in a 2×2 grid.

### Stage 2: Receptive Field Scaling

To provide Stage 3 convolution with its required 2×2 output, Stage 2 must generate a sufficient input patch. Using the relation: output size = n − f + 1 (where n = input size, f = filter size):

```
Stage 3 output needed: 2×2
Stage 3 Conv input needed: 2 + 3 − 1 = 4×4
```

Therefore Stage 2 must produce a **4×4×64** patch.

- **LIF 2** must output a **4×4×64** patch.
- **MaxPool 2** receives **8×8×64** and reduces to 4×4×64.
- **Stage 2 Convolution** (3×3, stride 1, no padding) must produce 8×8×64, requiring a **10×10×32** input from Stage 1:

```
Stage 2 output needed: 8×8
Stage 2 Conv input needed: 8 + 3 − 1 = 10×10
```

### Stage 1: Input Frame Requirements

To provide Stage 2 with its required 10×10×32 input, Stage 1 must produce a 10×10×32 LIF output:

- **LIF 1** outputs **10×10×32** spikes.
- **MaxPool 1** receives **20×20×32** and reduces to 10×10×32.
- **Stage 1 Convolution** (5×5, stride 1, no padding) must produce 20×20×32, requiring a **24×24×1** input:

```
Stage 1 output needed: 20×20
Stage 1 Conv input needed: 20 + 5 − 1 = 24×24
```

The full 256×256 input frame is zero-padded to **260×260** (2 pixels on each side). The hardware extracts **24×24 crops** from this padded image at the appropriate stride positions to produce the 4×4 = 16 LIF3 output positions.

### Summary of Backtracking Derivation

| Stage | Operation | Output Size | Requires Input Size |
|-------|-----------|------------|---------------------|
| 3 | LIF3 / MaxPool3 (2×2) | **1×1×128** | 2×2×128 |
| 3 | CONV3 (3×3, no pad) | 2×2×128 | **4×4×64** |
| 2 | LIF2 / MaxPool2 (2×2) | **4×4×64** | 8×8×64 |
| 2 | CONV2 (3×3, no pad) | 8×8×64 | **10×10×32** |
| 1 | LIF1 / MaxPool1 (2×2) | **10×10×32** | 20×20×32 |
| 1 | CONV1 (5×5, no pad) | 20×20×32 | **24×24×1** |
| — | Padded input crop | — | 24×24 from 260×260 |

**Key result:** A single **24×24 crop** from the padded 260×260 input, processed through all three stages, produces exactly **one LIF3 output pixel** (across 128 channels). To fill the complete 4×4×128 LIF3 output map, 16 such crops are processed sequentially across 6 memory frames.

---

## 4. Stage 1: 5×5 Convolution Architecture

### Input and Filter Specification

Stage 1 performs a 5×5 convolution over a single input channel (grayscale) to produce 32 output feature channels. The relevant parameters are:

| Parameter | Value |
|-----------|-------|
| Input frame | 256×256 grayscale, padded to 260×260 |
| Input window per operation | 24×24 crop (from backtracking) |
| Conv kernel | 5×5 |
| Input channels | 1 |
| Output filters | 32 |
| Stride | 1 |
| Conv output (per crop) | 20×20×32 |
| After MaxPool 2×2 | 10×10×32 |
| LIF1 output | 10×10×32 spikes |

The hardware must multiply a 5×5 filter against a 5×5 window of the input image and accumulate the 25 products. This fundamental computation is handled by the **CONV25 unit**.

---

### 4.1 The CONV25 Unit

The CONV25 unit computes a full 5×5 convolution for a single output pixel: it multiplies each of the 25 input elements P[0]…P[24] by the corresponding filter weight Q[0]…Q[24], then sums all 25 products.

#### Naive Implementation Cost

A straightforward implementation requires:
- **25 multipliers** (one per P×Q pair)
- **24 adders** arranged in a reduction tree: 12 → 6 → 3 → 2 → 1

Total: **49 DSP units** per CONV25 instance.

#### DSP Cascade Optimization

The DSP48E2 slices in UltraScale+ support **cascaded MAC chains**: each DSP has three inputs (A, B, and an accumulator input from the previous DSP's output). By chaining DSPs, each subsequent slice adds a new multiply-accumulate without requiring a separate adder:

```
DSP_0:  0 + P[0]×Q[0]              → acc_0
DSP_1:  acc_0 + P[1]×Q[1]          → acc_1
DSP_2:  acc_1 + P[2]×Q[2]          → acc_2
 ...
DSP_8:  acc_7 + P[8]×Q[8]          → acc_8  (final partial sum)
```

Each DSP slice performs both multiplication and accumulation in a **single pipeline stage (1 delay unit)**. A chain of 9 cascaded DSPs accumulates 9 products in 9 delays — this is the **conv9 primitive** used in Stages 2 and 3.

#### CONV25 Built from Three conv9 Chains

To compute 25 products, three parallel 9-DSP cascaded chains are used. The first chain processes P[0]…P[8] (9 elements), the second processes P[9]…P[17], and the third processes P[18]…P[24] (7 active elements, with inputs P[25] and P[26] set to zero to fill the chain to 9):

```
Chain A (DSP_0…DSP_8):  P[0]×Q[0] + P[1]×Q[1] + … + P[8]×Q[8]   → sum_A
Chain B (DSP_0…DSP_8):  P[9]×Q[9] + … + P[17]×Q[17]               → sum_B
Chain C (DSP_0…DSP_8):  P[18]×Q[18] + … + P[24]×Q[24] + 0 + 0     → sum_C

Final adder (1 DSP):    sum_A + sum_B + sum_C → CONV25 output
```

The three chains run **in parallel**. The final addition of the three partial sums adds 1 delay, giving a total critical path of:

```
9 (cascade chain) + 1 (final adder) = 10 delays
```

This reduces the naive 49-DSP implementation to **27 DSPs per CONV25 unit** (3 × 9 = 27), while achieving the same result.

<div align="center">
 <img width="1191" height="671" alt="image" src="https://github.com/user-attachments/assets/2594e342-6901-4c5e-a79c-4f5537ff692e" />
</div>

The same 9-DSP cascaded chain (conv9) is directly reused in Stages 2 and 3, making CONV25 a natural extension of the shared conv9 primitive rather than a separate design.

---

### 4.2 The Shaaban Unit: Backend Processing

After each CONV25 computes one dot product (one output pixel, one filter), the result must pass through Bias & ReLU, Batch Normalization, Max Pooling, and the LIF neuron. These operations are unified into a single hardware block called the **Shaaban unit**.

#### Why 4 CONV25 Inputs?

The Max Pooling layer uses a **2×2 window**: it selects the maximum value from 4 neighbouring spatial positions and outputs a single value. To generate **1 LIF output element**, the Shaaban unit must receive the **4 CONV25 outputs** corresponding to a 2×2 neighbourhood in the spatial output grid.

These 4 CONV25 units each receive a different 5×5 image window, where the 4 windows are spatially adjacent (forming a 2×2 arrangement in the output feature map), but all use the same filter weights.

#### Internal Pipeline

Each of the 4 inputs independently passes through its own Bias & ReLU and Batch Normalization blocks. The four results then enter a max-pool reduction tree and finally the LIF neuron:

```
Input[0] → Conv_Bias + ReLU → Batch Norm (A×x + B) ─┐
Input[1] → Conv_Bias + ReLU → Batch Norm (A×x + B) ─┤→ Max(0,1) ─┐
Input[2] → Conv_Bias + ReLU → Batch Norm (A×x + B) ─┤→ Max(2,3) ─┴→ Max → LIF → spike
Input[3] → Conv_Bias + ReLU → Batch Norm (A×x + B) ─┘
```

The full pipeline for the Shaaban unit consists of:

| Sub-block | Function | DSPs used |
|-----------|----------|-----------|
| 4× Conv_Bias + ReLU | Add trained bias, clamp negative values to 0 | 4 (normal adders) |
| 4× Batch Normalization | Fused affine: output = A×input + B | 4 MACs (DSP48E2) |
| 3× Max Pooling | 2-input comparators: 4→2→1 | 3 (comparators) |
| 1× LIF neuron | Integrate, decay, threshold, spike | 1 (3-adder DSP) |
| **Total** | | **13 DSPs** |

Critical path depth: **6 delays** (Bias+ReLU = 1, BN+Pool combined = 4, LIF = 1).

#### LIF Outputs

The LIF neuron produces two outputs:

- **MEM_OUT** — the updated membrane potential, written back to the same memory address as the input.
- **~Sign_Bit** — the spike output (inverted sign bit used as a 1-bit spike indicator), written to the spike memory at the same spatial position.

<div align="center">
 <img width="387" height="342" alt="shaban_unit" src="https://github.com/user-attachments/assets/e92b6e46-6bdf-4292-8cf3-795642d02cf5" />
</div>
---

### 4.3 Stage 1 Hardware: Combining CONV25 and Shaaban Units

#### One Filter, One Shaaban Unit

A single Shaaban unit produces **1 LIF1 output element** per cycle from 1 filter. It is driven by 4 CONV25 units, each processing one 5×5 image window from the 2×2 neighbourhood.

#### 32 Filters in Parallel

Stage 1 has 32 output filters. To process all 32 filters simultaneously, **32 Shaaban units** are instantiated (one per filter). Each Shaaban unit is driven by its own dedicated set of 4 CONV25 units, giving:

```
32 filters × 4 CONV25 units per filter = 128 CONV25 units total
32 Shaaban units (one per filter)
```

With all 32 filters computing in parallel, Stage 1 produces **32 output pixels per clock cycle**.

#### Cycles Required for Stage 1

From the backtracking analysis, Stage 1 must produce a **10×10×32 LIF1 output** (per crop). Since 32 channels are computed simultaneously in each cycle, the number of spatial positions determines the cycle count:

```
10 × 10 spatial positions = 100 cycles per 24×24 crop
```

---

### 4.4 Stage 1 Resource Summary

| Resource | Count / Value |
|----------|--------------|
| Conv units | 128 × CONV25 (4 per filter × 32 filters) |
| DSPs per CONV25 | 27 |
| Conv DSPs total | 128 × 27 = **3,456 DSPs** |
| Adder tree DSPs | 10 × 4 × 3 = **120 DSPs** |
| Extra ADD (conv25 final accumulation) | **8 DSPs** |
| Shaaban units | 32 (all active in Stage 1) |
| DSPs per Shaaban unit | 13 |
| Shaaban DSPs total | 32 × 13 = **416 DSPs** |
| **Total DSPs — Stage 1** | **4,000 DSPs (98.1% of 4,638)** |

### 4.5 Stage 1 Critical Path

| Path Segment | Delays |
|-------------|--------|
| CONV25 spatial convolution | 10 |
| Shaaban unit backend (Bias → BN → Pool → LIF) | 6 |
| **Total Stage 1 critical path** | **16 delays** |

---

## 5. Stage 2: 3×3 Convolution — 32 Input Channels

> *[This section will be completed in a future update.]*

---

## 6. Stage 3: 3×3 Convolution — 64 Input Channels

> *[This section will be completed in a future update.]*

---

## 7. Memory Control & Frame Mapping

> *[This section will be completed in a future update.]*

---

## 8. Classifier Head: GAP, FC1, FC2

> *[This section will be completed in a future update.]*

---

## 9. Full Resource Summary

> *[This section will be completed in a future update once all stages are documented.]*

| Stage | Conv DSPs | Adder Tree DSPs | Shaaban DSPs | Total DSPs | Utilization |
|-------|-----------|-----------------|--------------|------------|-------------|
| Stage 1 | 3,456 | 120 + 8 | 416 (32 active) | **4,000** | 98.1% |
| Stage 2 | 3,456 | 192 | 39 (3 active) | **3,687** | 90.5% |
| Stage 3 | 2,304 | 132 | 13 (1 active) | **2,449** | 60.1% |
| **Circuit Total** | — | — | — | **4,076** | **87.9%** |

Available DSP48E2 slices on XCVU11P: **4,638**
