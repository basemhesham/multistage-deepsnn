# deep_snn_top Block Diagram Summary

Diagram: [`deep_snn_top_block_diagram.drawio`](./deep_snn_top_block_diagram.drawio)

This diagram shows how `rtl/Top/top.sv` connects the implemented accelerator blocks. It intentionally stops at `spike_out`; the GAP/FC classifier head is planned for a larger wrapper and is not part of this top module.

## Main Data Paths

### Stage 1 Pixel Path

`pixel_mem` feeds the Stage 1 input mapping logic:

```text
pixel_mem -> Stage1_in / shift mapping -> pixels_s1 -> pixel MUX
```

This path is selected when `src_sel = 2'b00`.

### Stage 2 / Stage 3 Spike Path

`spike_mem` feeds the frame-based spike mapping logic:

```text
spike_mem -> mem_mapping -> frame_input_mapping x32 -> pixels_s2 -> pixel MUX
```

This path is selected when `src_sel = 2'b01` or `src_sel = 2'b10`.

### Weight Path

The three convolution weight maps are always available, then selected by `src_sel`:

```text
CONV1_W_MAP_OPT
CONV2_W_MAP_OPT  -> weight MUX -> weight unpacking -> conv9 array
CONV3_W_MAP_OPT
```

`conv2_filter` and `conv3_filter` select the active output filter for Stage 2 and Stage 3.

### Shared Compute Path

Both input paths and the selected weight path meet at the shared convolution hardware:

```text
pixel MUX -> conv9 array -> truncate -> adder_tree_shaaban_connect -> shaban_unit_top x32 -> spike_out
```

The same `conv9` array, adder tree, and Shaaban units are reused for all stages.

### Writeback Path

The Shaaban spike outputs are also formatted for the next stage:

```text
shaban_unit_top x32 -> mem_maping_1_2 -> spike BRAM TODO -> spike_mem
```

The BRAM itself is not instantiated in `top.sv` yet. The diagram marks it as a TODO because `mem_mapped_internal` is generated but still needs a real memory write/read path.

## Control Signals

The current top module receives control externally:

- `src_sel`: selects Stage 1, Stage 2, or Stage 3 routing.
- `frame`: selects the Stage 2/3 frame mapping window.
- `stage_sel`: selects the writeback layout in `mem_maping_1_2`.
- `conv2_filter`, `conv3_filter`: select Stage 2/3 weight filters.
- `enable`, `clk`, `rst`, `arst_n`: global timing/control inputs.

A future controller should generate these signals internally.

## Related RTL Blocks

| Diagram Block | RTL Location |
|---|---|
| Top integration | `rtl/Top/top.sv` |
| Stage 1 input mapping | `rtl/Stage 1/Stage1_in.sv` |
| Stage 2/3 memory mapping | `rtl/frame/frame_mapping_iterations_filters.sv` |
| Stage 2/3 frame mapping | `rtl/frame/frame_input_mapping_brackets.sv` |
| Conv weight maps | `rtl/SNN Parameters/Final Parameters (RTL)/CONV*_W_MAP_OPT.sv` |
| `conv9` array element | `rtl/convolution_blocks/conv9.sv` |
| Adder tree to Shaaban routing | `rtl/Shabaan_Adder_connect/adder_tree_shaaban_connect.sv` |
| Shaaban unit | `rtl/Shabaan Unit/shaban_unit_top.v` |
| Spike writeback mapping | `rtl/Controller/mem_maping_1_2.sv` |

