# Top Controller

`top_controller.sv` is the sequencing controller for the shared MultiStage-DeepSNN datapath. It owns the stage select, frame select, writeback layout select, convolution filter selectors, memory enables, padding clear control, and completion flag.

The controller is instantiated inside `rtl/Top/top.sv`, so these control signals are no longer external `deep_snn_top` inputs:

| Signal | Drives | Purpose |
|---|---|---|
| `stage` | `src_sel` inside `deep_snn_top` | Selects Stage 1, Stage 2, or Stage 3 datapath routing |
| `frame` | `mem_mapping` and `frame_input_mapping` | Cycles Stage 2 spatial frames `1..6` |
| `stage_sel` | `mem_maping_1_2` | Selects Stage 1-to-2 or Stage 2-to-3 writeback layout |
| `conv2_filter` | `CONV2_W_MAP_OPT` | Selects Stage 2 output filter `0..63` |
| `conv3_filter` | `CONV3_W_MAP_OPT` | Selects Stage 3 output filter `0..127` |
| `mem_enable` | future spike memory write/reset enables | Selects which memory bits are written or reset |
| `rd_enable` | future spike memory read enable | Enables readback for Stage 2/3 |
| `rd_mem_adderss` | future spike memory read address | Selects source memory bank |
| `wr_mem_adderss` | future spike memory write address | Selects destination memory bank |
| `zero_sel` | future padding/reset path | Indicates padding clear operation |
| `padding_flag` | future memory-bit reset control | Asserted while clearing padded destination memory |
| `done` | top-level status | Asserted after all configured fragments and temporal frames complete |

## FSM

The controller uses this state sequence:

```text
IDLE
  -> CLEAR_STAGE2_WORD
  -> STAGE1
  -> CLEAR_STAGE3_WORD
  -> STAGE2
  -> STAGE3
  -> DONE
```

`IDLE` waits for `enable`.

`CLEAR_STAGE2_WORD` prepares the 3200-bit Stage 2 input word before Stage 1 writes real data. It asserts `padding_flag`, `zero_sel`, and all `mem_enable` bits for the Stage 2 destination bank.

`STAGE1` runs 100 cycles for the 10x10 Stage 1 output positions. Each cycle enables one 32-bit group:

```text
stage1_pos 0  -> mem_enable[0 +: 32]
stage1_pos 1  -> mem_enable[32 +: 32]
...
stage1_pos 99 -> mem_enable[3168 +: 32]
```

`CLEAR_STAGE3_WORD` prepares the Stage 3 input word before Stage 2 writes real data. It also asserts `padding_flag`, `zero_sel`, and all `mem_enable` bits, but targets the Stage 3 destination bank.

`STAGE2` cycles `frame = 1..6` for each `conv2_filter = 0..63`. This matches the Stage 2 writeback layout used by `mem_maping_1_2`: each filter owns 16 spatial locations. Frames `1..5` enable three locations each, and frame `6` enables the final edge location.

`STAGE3` cycles `conv3_filter = 0..127`. Each cycle enables one final output-filter location.

`DONE` holds `done = 1` until `enable` is deasserted.

## Padding

Padding is handled by clearing the destination memory word before writing real outputs. Because the memory bits have their own reset, the padding states should drive the memory-bit reset path, not the global reset.

Recommended memory connection:

```systemverilog
bit_reset = padding_flag && mem_enable[i];
```

Do not use `rst` or `arst_n` for padding. Those are global controller/datapath resets. Padding clear only resets the selected destination memory bank so that lanes not written by Stage 1 or Stage 2 remain zero.

This matches the required layouts:

| Stage write | Destination layout | Padding behavior |
|---|---|---|
| Stage 1 output | 100 positions x 32 channels = 3200 bits | Clear all 3200 bits first, then write 100 real 32-bit groups |
| Stage 2 output | 64 filters x 16 positions = 1024 used bits | Clear destination first, then write only the 4x4x64 Stage 3 input positions |
| Stage 3 output | 128 final filter bits | No padded intermediate word is currently needed after Stage 3 |

## Top Integration

`deep_snn_top` instantiates the controller as `u_top_controller`. The controller output `stage` is connected internally as `src_sel`, which drives:

- the pixel source MUX,
- the weight ROM MUX,
- the adder-tree-to-Shaaban routing.

The controller output `stage_sel` drives `mem_maping_1_2`, so Stage 1 and Stage 2 writebacks use the correct packed memory layout.

The BRAM itself is still future work in `top.sv`. The controller already provides the write/read enables, addresses, and padding reset controls that the BRAM wrapper should consume.
