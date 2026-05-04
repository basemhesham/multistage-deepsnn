# Module Specification: Adder Tree → Shaaban Unit Connection Layer

**Project:** MultiStage-DeepSNN CNN-SNN Hybrid FPGA Accelerator  
**File:** `adder_tree_shaaban_connect.sv`  
**Sub-module:** `ext_sum_correction.sv`  
**Status:** RTL Complete — Pending Simulation Sign-off  

---

## 1. Purpose

This module is the **datapath bridge between the adder tree array and the Shaaban (conv-BN-ReLU-pool-LIF) units**. It collects partial sums from 12 adder trees, assembles them correctly into 128 inputs, and distributes them across 32 Shaaban units — all while supporting three pipeline stages through a single shared hardware pool.

---

## 2. File List

| File | Role |
|---|---|
| `adder_tree_shaaban_connect.sv` | Top-level connection and MUX |
| `ext_sum_correction.sv` | Orphan MAC correction adders (sub-module) |
| `adder_tree_10_4_1_1.v` | Adder tree (pre-existing, need to be modified) |
| `shaban_unit_top.v` | Shaaban unit (pre-existing, not modified) |
| `tb_adder_tree_shaaban_connect.sv` | Testbench (T1–T6) |

---

## 3. Interface (adder_tree_shaaban_connect)

| Port | Direction | Width | Description |
|---|---|---|---|
| `clk` | in | 1 | System clock |
| `rst` | in | 1 | Synchronous active-high reset |
| `src_sel` | in | 2 | Stage selector (see Section 5) |
| `mac_in` | in | 12×32×18 | All MAC products from conv units |
| `final_s3` | in | 18 | Pre-accumulated Stage 3 sum |
| `conv_bias` | in | 18 | Shared bias for all Shaaban units |
| `mult_weight` | in | 18 | Batch-norm multiply weight |
| `add_weight` | in | 18 | Batch-norm add weight |
| `spike_out` | out | 32 | One spike bit per Shaaban unit |

---

## 4. Parameters

| Parameter | Default | Description |
|---|---|---|
| `N_TREES` | 12 | Number of adder tree instances |
| `TAPS_PER_TREE` | 10 | L1 tap outputs per tree (conv25_1..10) |
| `N_SHAABAN` | 32 | Number of Shaaban unit instances |
| `INPUTS_PER_SHB` | 4 | Inputs per Shaaban (conv_bias_relu_num) |
| `POOL_NUM` | 2 | Pooling inputs per Shaaban |
| `DATA_WIDTH` | 18 | Signed fixed-point bit width |
| `N_CORRECTION` | 8 | Derived: 128 − 120 orphan corrections |

> **Rule:** Only change `N_TREES`, `TAPS_PER_TREE`, `N_SHAABAN`, `INPUTS_PER_SHB`, `DATA_WIDTH`. All others auto-derive.

---

## 5. Stage Routing (src_sel)

The 32 Shaaban units are **always physically present** but only receive valid data in their assigned stage. Inactive units receive `conv_in = 0` and produce no spike.

| src_sel | Stage | Active Shaabans | Input Source |
|---|---|---|---|
| `2'b00` | Stage 1 | All 32 | `flat_s1[s*4 .. s*4+3]` |
| `2'b01` | Stage 2 | 0, 1, 2 | `tree_final[s*4 .. s*4+3]` |
| `2'b10` | Stage 3 | 0 only | `final_s3` at slot 0 |

---

## 6. Stage 1 Assembly — The Core Problem and Solution

### Why 120 taps are not enough

Each adder tree takes 32 MAC inputs and groups them into 10 Layer-1 adders of 3 inputs each (using slots 0–29). This produces 10 complete conv25 partial sums as L1 tap outputs.

```
12 trees × 10 taps = 120 direct conv25 results
32 Shaabans × 4 inputs = 128 inputs needed
Gap = 8 missing inputs
```

### What happens to slots 30 and 31 of each tree

Slots 30 and 31 of every tree (the "orphans") are fed into the tree's internal DSP macro via a different path. Due to a bit-range issue in `final_output`, these cannot be reliably used directly. Instead, **all 24 orphan values (12 trees × 2) are collected externally** and re-grouped by the `ext_sum_correction` module.

### Orphan correction grouping

The 24 orphans form a flat pool indexed 0–23, where `pool[i] = mac_in[i/2][30 + i%2]`. They are grouped sequentially into 8 three-input adders:

```
corr[0]: tree0[30], tree0[31], tree1[30]
corr[1]: tree1[31], tree2[30], tree2[31]
corr[2]: tree3[30], tree3[31], tree4[30]
corr[3]: tree4[31], tree5[30], tree5[31]
corr[4]: tree6[30], tree6[31], tree7[30]
corr[5]: tree7[31], tree8[30], tree8[31]
corr[6]: tree9[30], tree9[31], tree10[30]
corr[7]: tree10[31], tree11[30], tree11[31]
```

### flat_s1 layout (128 entries)

```
Trees 0–7  (stride 11): [t*11 .. t*11+9] = taps[0..9]
                         [t*11 + 10]      = corr_out[t]
Trees 8–11 (stride 10): [88+(t-8)*10 .. +9] = taps[0..9]

Total = (8 × 11) + (4 × 10) = 88 + 40 = 128 ✓
```

Each Shaaban unit `s` receives `flat_s1[s*4 .. s*4+3]`.

---

## 7. Sub-module: ext_sum_correction

**File:** `ext_sum_correction.sv`  
**Type:** Purely combinational — no clock or reset.

| Port | Direction | Width | Description |
|---|---|---|---|
| `mac_in` | in | 12×32×18 | Full MAC array (only [30] and [31] are read) |
| `corr_out` | out | 8×18 | 8 corrected partial sums |

**Bit-width handling:**  
Three 18-bit signed values are sign-extended to 20 bits before addition (prevents overflow). The 20-bit raw sum is right-shifted by 1 (truncate LSB) to match the normalization applied to L1 taps inside the adder tree. Output is 18 bits.

---

## 8. Testbench Coverage (T1–T6)

| Test | What it drives | What it checks |
|---|---|---|
| T1 | Reset | spike_out === 0 after reset |
| T2 | Stage 1, all MACs = 80 | Some spikes from all 32 Shaabans |
| T3 | Stage 2, all MACs = 100 | spike_out[31:3] === 0 |
| T4 | Stage 3, final_s3 = 200 | spike_out[31:1] === 0 |
| T5 | Stage 1→2 switch mid-run | Units 3..31 go silent after switch |
| T6 | Only tree0[30,31] + tree1[30] driven | Shaabans 0,1 silent; Shaaban 2 may spike |

---

## 9. Synthesis Notes

- All combinational paths (MUX, correction adders, flat_s1 assembly) are **zero pipeline stages** — timing closure is the caller's responsibility.
- `unique case` on `src_sel` tells synthesis there are no overlapping cases and enables priority-free mux inference.
- The `generate` blocks are fully parametric — changing `N_TREES` or `N_SHAABAN` rebuilds the entire connection automatically.
- Target: Xilinx Vivado (UltraScale+). The correction adders should infer LUT-based adders; force DSP48E2 with `use_dsp = yes` attribute if timing requires it.

---

## 10. Known Limitations / Open Questions

1. **Per-filter weight ROM addressing** is handled outside this module. The current shared `conv_bias`, `mult_weight`, `add_weight` ports assume all Shaabans use the same weights in a given cycle — revise if per-unit weight banks are added.
2. **Stage 2 tree_final indexing** assumes `s * INPUTS_PER_SHB + p < N_TREES` for active Shaabans (s=0,1,2 × 4 = 0..11; 12 trees available — OK).
3. **final_s3** for Stage 3 is a single scalar. If multiple spatial positions are needed simultaneously, this port must be widened.
