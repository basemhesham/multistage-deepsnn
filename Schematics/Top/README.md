# deep_snn_top Block Diagram Summary

Diagram: [`deep_snn_top_block_diagram.drawio`](./deep_snn_top_block_diagram.drawio)

This document explains the blocks shown in the `deep_snn_top` diagram and links each implemented block to its RTL source. The top module is the shared convolution/Shaaban datapath for the three SNN stages. It intentionally stops at `spike_out`; the final classifier head is planned for a larger wrapper.

Top RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

## Big Picture

`deep_snn_top` has two input data paths and one shared compute path:

```text
Stage 1 path:
pixel_mem -> Stage1_in shift mapping -> pixels_s1 -> pixel MUX

Stage 2/3 path:
spike_mem -> mem_mapping -> frame_input_mapping x32 -> pixels_s2 -> pixel MUX

Shared compute path:
pixel MUX -> conv9 array -> truncate -> adder_tree_shaaban_connect -> shaban_unit_top x32 -> spike_out
```

The same physical `conv9` array, adder-tree routing, and Shaaban units are reused for Stage 1, Stage 2, and Stage 3. The controller selects the active stage using `src_sel`.

## Block Details

### `pixel_mem`

Status: interface exists in `top.sv`; real pixel BRAM is future work.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

`pixel_mem` is a flat input bus for Stage 1 raw image pixels:

```text
pixel_mem[6911:0] = 384 pixels x 18 bits
```

It is unpacked into `in_mem[384]`, then sent to the Stage 1 shift mapping logic. It is selected only when `src_sel = 2'b00`.

### `Stage1_in` / Shift Mapping

Status: implemented inside `top.sv`; related standalone RTL exists.

RTL:

- Integration in [`rtl/Top/top.sv`](../../rtl/Top/top.sv)
- Standalone reference: [`rtl/Stage 1/Stage1_in.sv`](../../rtl/Stage%201/Stage1_in.sv)

Function:

```text
in_mem[384] -> p_imag[384] -> pixels_s1[12][32][9]
```

Stage 1 uses `conv9` units to build 5x5 convolution results. Since three `conv9` outputs are grouped to make one CONV25-equivalent result, each 32-unit block leaves two orphan positions. The shift mapping rearranges the Stage 1 pixel order so these groups line up correctly across the 12 blocks.

The final mapping from `p_imag` to `pixels_s1` is:

```systemverilog
pixels_s1[g][c][t] = p_imag[g * 32 + c]
```

where `g = 0..11`, `c = 0..31`, and `t = 0..8`.

### `spike_mem`

Status: interface exists in `top.sv`; real spike BRAM is future work.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

`spike_mem` is a flat input bus for Stage 2 and Stage 3 spike data:

```text
spike_mem[3199:0] = 3200 one-bit spikes
```

It represents spikes written by the previous stage. Stage 2 reads Stage 1 spikes; Stage 3 reads Stage 2 spikes. In the current top, this is still an external input bus. The internal BRAM write/read path is not implemented yet.

### `mem_mapping`

Status: implemented.

RTL: [`rtl/frame/frame_mapping_iterations_filters.sv`](../../rtl/frame/frame_mapping_iterations_filters.sv)

Inputs:

```text
spike_mem[3199:0]
frame
```

Output:

```text
fil_in[32][40]
```

Function: for the current `frame`, select the correct spike bits from the flat `spike_mem` bus and arrange them into 32 groups of 40 bits.

Example behavior:

```systemverilog
case(frame)
  1: fil_in[0][0] = mem[0];
  2: fil_in[0][0] = mem[192];
  3: fil_in[0][0] = mem[768];
  ...
endcase
```

So `frame` controls which spatial region of the stored spike map is routed to the convolution input logic.

### `frame_input_mapping x32`

Status: implemented.

RTL: [`rtl/frame/frame_input_mapping_brackets.sv`](../../rtl/frame/frame_input_mapping_brackets.sv)

There are 32 generated instances in `top.sv`, one per channel/filter:

```text
fil_in[0][40]  -> frame_input_mapping -> conv_windows[0][12][9]
...
fil_in[31][40] -> frame_input_mapping -> conv_windows[31][12][9]
```

Function: convert each 40-bit selected spike group into 12 parallel 3x3 windows:

```text
fil_in[40] -> conv[12][9]
```

Then `top.sv` transposes/sign-extends the result into:

```text
pixels_s2[12][32][9]
```

This becomes the Stage 2/3 input to the shared `conv9` array.

### Pixel MUX

Status: implemented inside `top.sv`.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

Inputs:

```text
pixels_s1[12][32][9]
pixels_s2[12][32][9]
src_sel
```

Output:

```text
pixels_mapped[12][32][9]
```

Function: choose which pixel/spike input path feeds the shared `conv9` array.

| `src_sel` | Selected Input | Stage |
|---|---|---|
| `2'b00` | `pixels_s1` | Stage 1 |
| `2'b01` | `pixels_s2` | Stage 2 |
| `2'b10` | `pixels_s2` | Stage 3 |

This is a replicated MUX over all `12 x 32 x 9 = 3456` input positions.

### Convolution Weight Maps

Status: implemented.

RTL:

- [`CONV1_W_MAP_OPT.sv`](../../rtl/SNN%20Parameters/Final%20Parameters%20%28RTL%29/CONV1_W_MAP_OPT.sv)
- [`CONV2_W_MAP_OPT.sv`](../../rtl/SNN%20Parameters/Final%20Parameters%20%28RTL%29/CONV2_W_MAP_OPT.sv)
- [`CONV3_W_MAP_OPT.sv`](../../rtl/SNN%20Parameters/Final%20Parameters%20%28RTL%29/CONV3_W_MAP_OPT.sv)

These blocks provide the trained convolution weights in the exact order expected by the `conv9` array.

Outputs:

```text
stage1_weights[3456]
stage2_weights[3456]
stage3_weights[3456]
```

Why 3456?

```text
384 conv9 units x 9 weights = 3456 weights
```

Stage 2 and Stage 3 also receive filter selectors:

```text
conv2_filter -> CONV2_W_MAP_OPT
conv3_filter -> CONV3_W_MAP_OPT
```

### Weight MUX

Status: implemented inside `top.sv`.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

Inputs:

```text
stage1_weights[3456]
stage2_weights[3456]
stage3_weights[3456]
src_sel
```

Output:

```text
active_weights[3456]
```

Function: select the weight set for the active stage.

| `src_sel` | Selected Weights |
|---|---|
| `2'b00` | `stage1_weights` |
| `2'b01` | `stage2_weights` |
| `2'b10` | `stage3_weights` |

### Weight Unpacking

Status: implemented inside `top.sv`.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

Input:

```text
active_weights[3456]
```

Output:

```text
weights_mapped[12][32][9]
```

Function: reshape the flat selected weight list into the shape needed by the generated `conv9` array:

```systemverilog
weights_mapped[g][c][t] = active_weights[(g * 32 + c) * 9 + t]
```

### `conv9 array`

Status: implemented.

RTL:

- Array generated in [`rtl/Top/top.sv`](../../rtl/Top/top.sv)
- Element module: [`rtl/convolution_blocks/conv9.sv`](../../rtl/convolution_blocks/conv9.sv)
- DSP wrapper: [`rtl/convolution_blocks/convDspAddMult.v`](../../rtl/convolution_blocks/convDspAddMult.v)

The top instantiates:

```text
12 rows x 32 columns = 384 conv9 units
```

Each `conv9` receives:

```text
P[0:8] = pixels_mapped[g][c][0:8]
Q[0:8] = weights_mapped[g][c][0:8]
```

and produces:

```text
mac_raw[g][c]  // 40-bit dot product
```

Internally, one `conv9` is a 9-DSP48E2 cascaded MAC chain.

### Truncate

Status: implemented inside `top.sv`.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

Input:

```text
mac_raw[12][32]  // 40-bit per value
```

Output:

```text
mac_to_connect[12][32]  // 18-bit per value
```

Function: convert the 40-bit `conv9` result to the 18-bit Q7.10 datapath width used by the adder tree and Shaaban units:

```systemverilog
mac_to_connect[g][c] = mac_raw[g][c][17:0]
```

### `adder_tree_shaaban_connect`

Status: implemented.

RTL:

- [`rtl/Shabaan_Adder_connect/adder_tree_shaaban_connect.sv`](../../rtl/Shabaan_Adder_connect/adder_tree_shaaban_connect.sv)
- [`rtl/Shabaan_Adder_connect/ext_sum_correction.sv`](../../rtl/Shabaan_Adder_connect/ext_sum_correction.sv)
- Adder layers: [`rtl/adder tree`](../../rtl/adder%20tree/)

Inputs:

```text
mac_to_connect[12][32]
src_sel
```

Output:

```text
shb_bus[32]
```

Function: combine and route `conv9` outputs to the Shaaban units.

Stage behavior:

| `src_sel` | Function |
|---|---|
| `2'b00` | Stage 1: group `conv9` outputs into CONV25-equivalent results and feed all 32 Shaaban units |
| `2'b01` | Stage 2: use full tree sums and feed Shaaban units 0, 1, and 2 |
| `2'b10` | Stage 3: route the required accumulated value to Shaaban unit 0 |

This block is the bridge between the shared convolution array and the 32 Shaaban units.

### `shaban_unit_top x32`

Status: implemented.

RTL:

- [`rtl/Shabaan Unit/shaban_unit_top.v`](../../rtl/Shabaan%20Unit/shaban_unit_top.v)
- [`conv_bias_Relu.v`](../../rtl/Shabaan%20Unit/conv_bias_Relu.v)
- [`Batch_Norm.v`](../../rtl/Shabaan%20Unit/Batch_Norm.v)
- [`Max_pooling.v`](../../rtl/Shabaan%20Unit/Max_pooling.v)
- [`LIF.v`](../../rtl/Shabaan%20Unit/LIF.v)

Inputs:

```text
shb_bus[s]          // 4 packed 18-bit convolution values
conv_bias
mult_weight
add_weight
clk, rst
```

Output:

```text
spike_out[s]
```

There are 32 Shaaban units physically instantiated. Depending on the stage, only some receive useful nonzero inputs:

| Stage | Active Shaaban Units |
|---|---|
| Stage 1 | 32 units |
| Stage 2 | 3 units |
| Stage 3 | 1 unit |

Each Shaaban unit performs:

```text
conv_bias_Relu x4 -> Batch_Norm x4 -> MaxPool 4->2->1 -> LIF -> spike
```

### `spike_out`

Status: implemented top-level output.

RTL: [`rtl/Top/top.sv`](../../rtl/Top/top.sv)

Output:

```text
spike_out[31:0]
```

This is the visible top-level spike output. Valid bits depend on the active stage:

| Stage | Valid Output Bits |
|---|---|
| Stage 1 | `spike_out[31:0]` |
| Stage 2 | `spike_out[2:0]` |
| Stage 3 | `spike_out[0]` |

### `mem_maping_1_2`

Status: implemented, but its output is not connected to a real BRAM yet.

RTL: [`rtl/Controller/mem_maping_1_2.sv`](../../rtl/Controller/mem_maping_1_2.sv)

Inputs:

```text
stage_sel
shaaban_spike_bus[32]
```

Output:

```text
mem_mapped_internal[3200]
```

Function: format Shaaban spike outputs into the flat memory layout needed by the next stage.

| `stage_sel` | Writeback Layout |
|---|---|
| `1'b0` | Stage 1 output -> Stage 2 input layout |
| `1'b1` | Stage 2 output -> Stage 3 input layout |

Current limitation: `mem_mapped_internal` is generated internally, but no BRAM is instantiated yet to store and read it back as `spike_mem`.

## Control Signals

Current status: external inputs to `deep_snn_top`; future controller should generate them.

| Signal | Current Source | Function |
|---|---|---|
| `src_sel` | External/testbench | Selects Stage 1, Stage 2, or Stage 3 routing |
| `frame` | External/testbench | Selects the Stage 2/3 memory window |
| `stage_sel` | External/testbench | Selects writeback layout in `mem_maping_1_2` |
| `conv2_filter` | External/testbench | Selects Stage 2 output filter weights |
| `conv3_filter` | External/testbench | Selects Stage 3 output filter weights |
| `enable` | External/testbench | Enables mapping/control-dependent logic |
| `clk`, `rst`, `arst_n` | External/testbench/system | Timing and reset |

## What Already Exists

These blocks are implemented and connected inside `deep_snn_top`:

| Block | Status |
|---|---|
| Stage 1 pixel unpacking and shift mapping | Connected in top |
| Stage 2/3 `mem_mapping` | Connected in top |
| `frame_input_mapping x32` | Connected in top |
| Pixel MUX | Connected in top |
| `CONV1_W_MAP_OPT`, `CONV2_W_MAP_OPT`, `CONV3_W_MAP_OPT` | Connected in top |
| Weight MUX and weight unpacking | Connected in top |
| `conv9` array, 384 units | Connected in top |
| MAC truncation | Connected in top |
| `adder_tree_shaaban_connect` | Connected in top |
| `shaban_unit_top x32` | Connected in top |
| `mem_maping_1_2` writeback formatter | Connected in top, output awaiting BRAM |

These RTL blocks exist but are not part of `deep_snn_top` yet:

| Block | RTL |
|---|---|
| FC1 layer | [`rtl/classifier_head/fc1_layer.sv`](../../rtl/classifier_head/fc1_layer.sv) |
| FC2 layer | [`rtl/classifier_head/fc2_layer.sv`](../../rtl/classifier_head/fc2_layer.sv) |
| FC golden model/tests | [`rtl/classifier_head`](../../rtl/classifier_head/) |

## Future Work Blocks

These blocks are not fully implemented/connected in `deep_snn_top` yet:

| Future Block | Needed Function |
|---|---|
| Controller FSM | Generate `src_sel`, `frame`, `stage_sel`, filter selectors, write enables, counters, and temporal sequencing |
| Spike BRAM | Store `mem_mapped_internal` and feed it back as `spike_mem` for Stage 2/3 |
| Pixel BRAM/read interface | Feed `pixel_mem` from real image/crop memory instead of an external flat bus |
| BRAM address/write control | Provide write address, read address, and valid/write-enable timing for spike memory |
| Temporal loop controller | Repeat the pipeline across `T = 16` frames and preserve LIF membrane state correctly |
| GAP block | Convert final Stage 3 spikes into the 128-input classifier vector |
| Classifier wrapper | Connect `spike_out`/GAP output to FC1 and FC2 |
| FC integration | Instantiate existing `fc1_layer` and `fc2_layer` in the larger wrapper |
| Final class output interface | Provide class logits or argmax result to the system |

