// =============================================================================
// FILE       : deep_snn_top.sv
// PROJECT    : MultiStage-DeepSNN FPGA Accelerator
// TARGET     : Xilinx Virtex UltraScale+ XCVU11P (flga2577-3-e package)
// CLOCK      : 40 MHz
// DATA FMT   : 18-bit Q7.10 signed fixed-point
// AUTHORS    : [Your team names here]
// =============================================================================
//
// PURPOSE
// -------
// This is the top-level integration module for the MultiStage-DeepSNN
// accelerator. It connects all sub-modules and defines the complete data flow
// from input memory through convolution, accumulation, and spike generation,
// and finally back to memory for the next stage.
//
// The accelerator classifies driving incidents from dashcam video into 4 classes:
//   0 = no incident
//   1 = drifting/skidding
//   2 = other crash
//   3 = collision
//
// =============================================================================
// THE CORE HARDWARE INNOVATION: TIME-MULTIPLEXED SHARED CIRCUIT
// =============================================================================
//
// Instead of building separate convolution hardware for each of the 3 SNN stages,
// ONE physical circuit is reused across all stages. A 2-bit control signal
// (src_sel) switches the routing of inputs and outputs at each stage boundary:
//
//   src_sel = 2'b00  -->  Stage 1 active  (5x5 conv, all 32 Shaabans)
//   src_sel = 2'b01  -->  Stage 2 active  (3x3 conv, Shaabans 0,1,2 only)
//   src_sel = 2'b10  -->  Stage 3 active  (3x3 conv, Shaaban 0 only)
//
// This means 384 conv9 units, 32 Shaaban units, and all adder trees are always
// physically present on the FPGA. Inactive units simply receive zero inputs
// and consume no dynamic power.
//
// Why this works: The three stages process different spatial sizes:
//   Stage 1: 24x24 crop  -->  20x20 output  -->  10x10 after MaxPool
//   Stage 2: 10x10       -->   8x8  output  -->   4x4  after MaxPool
//   Stage 3:  4x4        -->   2x2  output  -->   1x1  after MaxPool
// As stages progress, fewer output pixels are needed, so fewer Shaabans are used.
//
// =============================================================================
// COMPLETE DATA FLOW
// =============================================================================
//
// TWO COMPLETELY SEPARATE INPUT PATHS feed the shared conv9 array:
//
//  +---------------------------------------------------------------------------+
//  | PATH A: STAGE 1 INPUT                                                     |
//  | Source: Raw grayscale pixel BRAM (18-bit fixed-point pixel values)        |
//  |                                                                           |
//  |  pixel_mem[6911:0]   -- 384 x 18-bit packed pixel words                  |
//  |       |                                                                   |
//  |       v  (A1: unpack into array)                                         |
//  |  in_mem[384]         -- 384 signed 18-bit pixel words                    |
//  |       |                                                                   |
//  |       v  (A2: Stage1_in shift-mapping)                                   |
//  |  p_imag[384]         -- shift-corrected pixel stream                     |
//  |                                                                           |
//  |  WHY THE SHIFT? The conv9 array has 12 blocks of 32 units each.          |
//  |  Each CONV25 needs 3 consecutive conv9 units. 32/3 = 10 groups of 3,     |
//  |  leaving 2 "orphan" conv9 units at the end of each block that cannot     |
//  |  form a complete group within that block.                                 |
//  |  The shift-mapping pre-routes pixels so orphans from block B pair with   |
//  |  the start of block B+1, forming complete groups across block boundaries.|
//  |  The pattern cycles every 3 blocks: 0-shift, 1-shift, 2-shift, repeat.  |
//  |       |                                                                   |
//  |       v  (A3: broadcast to 9 taps)                                       |
//  |  pixels_s1[12][32][9]   -- ready for conv9 array                        |
//  +---------------------------------------------------------------------------+
//
//  +---------------------------------------------------------------------------+
//  | PATH B: STAGE 2 INPUT                                                     |
//  | Source: Spike output BRAM (1-bit spikes written by Stage 1 writeback)     |
//  |                                                                           |
//  |  spike_mem[3199:0]   -- 3200 x 1-bit spike values                        |
//  |       |                                                                   |
//  |       v  (B1: mem_mapping, frame_mapping_iterations_filters.sv)          |
//  |  fil_in[32][40]      -- 32 filters x 40 spike positions per frame        |
//  |                                                                           |
//  |  WHY frame? Stage 2 cannot read all 3200 spikes at once into the         |
//  |  12 conv engines. Instead the "frame" signal (1-6) acts as a sliding     |
//  |  window pointer, re-routing different subsets of spikes to the engines   |
//  |  each cycle. This implements a virtual sliding window with zero data-     |
//  |  movement cost -- only MUX select lines change, not data registers.      |
//  |       |                                                                   |
//  |       v  (B2: frame_input_mapping x32, frame_input_mapping_brackets.sv) |
//  |  conv_windows[32][12][9]  -- 32 filters x 12 engines x 9 taps (1-bit)   |
//  |       |                                                                   |
//  |       v  (B3: sign-extend 1-bit to 18-bit)                               |
//  |  pixels_s2[12][32][9]   -- ready for conv9 array                        |
//  +---------------------------------------------------------------------------+
//
//  +---------------------------------------------------------------------------+
//  | PATH C: STAGE 3 INPUT                                                     |
//  | Source: Stage 2 spike writeback layout                                    |
//  |                                                                           |
//  |  spike_mem[1023:0]   -- 64 channels x 16 positions = 4x4x64              |
//  |       |                                                                   |
//  |       v  (C1: bin_muxing_stage2)                                         |
//  |  stage3_windows[9][4][64] -- 9 taps x 4 windows x 64 channels           |
//  |       |                                                                   |
//  |       v  (C2: split 64 channels into adjacent 32-channel row pairs)      |
//  |  pixels_s3[12][32][9]  -- rows 0..7 active, rows 8..11 zero             |
//  +---------------------------------------------------------------------------+
//
//  +---------------------------------------------------------------------------+
//  | SHARED HARDWARE (both paths feed into this)                               |
//  |                                                                           |
//  |  src_sel MUX                                                              |
//  |  pixels_mapped[12][32][9]  <-- pixels_s1 when src_sel=2'b00 (Stage 1)   |
//  |                             <-- pixels_s2 when src_sel=2'b01 (Stage 2)   |
//  |                             <-- pixels_s3 when src_sel=2'b10 (Stage 3)   |
//  |       |                                                                   |
//  |       |   Weight ROMs (distributed LUTs, always readable, zero latency)  |
//  |       |   CONV1_W_MAP_OPT --> stage1_weights[3456]                       |
//  |       |   CONV2_W_MAP_OPT --> stage2_weights[3456]                       |
//  |       |   CONV3_W_MAP_OPT --> stage3_weights[3456]                       |
//  |       |   src_sel MUX --> active_weights[3456]                           |
//  |       |                                                                   |
//  |       v                                                                   |
//  |  conv9 array [12 rows x 32 cols = 384 units]                             |
//  |  Each conv9: P[9] x Q[9] --> 40-bit MAC (9 cascaded DSP48E2)            |
//  |  Total DSPs for convolution: 384 x 9 = 3,456 DSP48E2                    |
//  |       |                                                                   |
//  |       v  (truncate 40-bit to 18-bit)                                     |
//  |  mac_to_connect[12][32]                                                   |
//  |       |                                                                   |
//  |       v                                                                   |
//  |  adder_tree_shaaban_connect                                               |
//  |  Internally contains:                                                     |
//  |    - 12 x adder_tree_10_4_1_1 (one per block row)                        |
//  |        Each tree: 32 conv9 inputs --> 10 Layer-1 partial sums (CONV25)   |
//  |                                   --> 1 full sum (Stage 2/3 path)        |
//  |    - ext_sum_correction: collects the 2 orphan outputs from each of      |
//  |        12 trees (24 orphan values total) and groups them into 8 three-   |
//  |        input correction sums (corr_out[0..7])                            |
//  |    - flat_s1[128] assembly:                                               |
//  |        Trees 0-7:  flat_s1[t*11 .. t*11+9] = tree_tap[t][0..9]          |
//  |                    flat_s1[t*11+10]         = corr_out[t]                |
//  |        Trees 8-11: flat_s1[88+(t-8)*10 ..+9] = tree_tap[t][0..9]        |
//  |        Total: 120 tree taps + 8 corrections = 128 values = 32x4          |
//  |    - 3-way src_sel MUX:                                                   |
//  |        2'b00: shb_bus[s] = flat_s1[s*4 .. s*4+3]  (Stage 1, all 32)    |
//  |        2'b01: shb_bus[0,1,2] = tree finals; shb_bus[3..31] = 0          |
//  |        2'b10: shb_bus[0] = four Stage 3 64-channel sums; rest = 0        |
//  |       |                                                                   |
//  |       v                                                                   |
//  |  shb_bus[32]  -- each entry is 4 packed 18-bit values                   |
//  |       |                                                                   |
//  |       v                                                                   |
//  |  shaaban_unit_top x32  (always present, inactive units get zero input)   |
//  |  Each unit: conv_bias_Relu x4 --> Batch_Norm x4 --> MaxPool(4->2->1)    |
//  |             --> LIF --> spike  (13 DSPs per unit)                         |
//  |  Active units per stage: Stage1=32, Stage2=3, Stage3=1                   |
//  |       |                                                                   |
//  |       v                                                                   |
//  |  spike_out[32]  -- 1-bit per unit                                         |
//  |       |                                                                   |
//  |       v  (write-back, INTERNAL only -- NOT a top-level port)             |
//  |  mem_maping_1_2                                                           |
//  |  Converts spikes --> 3200-element memory layout for next stage            |
//  |  mem_mapped_internal[3200] --> BRAM write port (TODO: connect to BRAM)   |
//  +---------------------------------------------------------------------------+
//
// =============================================================================
// RESOURCE SUMMARY (from README Section 5.5 and 10)
// =============================================================================
//
//   Conv9 units total:          384  (12 blocks x 32)
//   DSPs for convolution:     3,456  (384 x 9)
//   DSPs for adder trees:       120  (10 taps x 4 layers x 3 branches)
//   DSPs for orphan correction:   8
//   DSPs for Shaabans:           416  (32 x 13)
//   Total DSPs Stage 1:        4,000  (98.1% of 4,638 available)
//
// =============================================================================
// WHAT IS COMPLETE AND WHAT STILL NEEDS WORK
// =============================================================================
//
//   COMPLETE:
//   [x] Stage 1 input path (Stage1_in shift-mapping from pixel BRAM)
//   [x] Stage 2 input path (mem_mapping + frame_input_mapping)
//   [x] Stage 3 input path (bin_muxing_stage2 from Stage 2 writeback layout)
//   [x] src_sel pixel MUX (switches between Stage 1, Stage 2, and Stage 3)
//   [x] Stage 1 weight ROM (CONV1_W_MAP_OPT)
//   [x] Stage 2 weight ROM (CONV2_W_MAP_OPT)
//   [x] Stage 3 weight ROM (CONV3_W_MAP_OPT)
//   [x] src_sel weight MUX (switches active weight ROM)
//   [x] conv9 array (384 units, shared across all stages)
//   [x] adder_tree_shaaban_connect (summation + stage routing)
//   [x] 32 Shaaban units (always instantiated, inactive ones get zero)
//   [x] mem_maping_1_2 write-back converter
//   [x] top_controller control FSM
//
//   INCOMPLETE / TODO:
//   [ ] BRAM instantiation -- mem_mapped_internal currently goes nowhere.
//       Needs: Instantiate a BRAM primitive (RAMB36E2 or UltraRAM) here.
//       Connect: mem_mapped_internal --> BRAM write data port.
//       Connect the controller BRAM enables/address outputs to the memory.
//
//   [ ] Classifier head (GAP, FC1, FC2) -- not instantiated here yet.
//       After Stage 3 produces spikes, they feed into:
//         Global Average Pool (AdaptiveAvgPool 1x1)
//         FC1: 128 -> 256 + ReLU + Dropout
//         FC2: 256 -> 4 (class logits)
//       See README Section 9 for details.
//
//   [ ] Temporal loop (T=16 frames) -- the LIF neurons accumulate membrane
//       potential across 16 temporal input frames. This top module processes
//       one temporal frame at a time. The controller must loop T=16 times
//       before reading the final spike output.
//
// =============================================================================

module deep_snn_top #(
    // -------------------------------------------------------------------------
    // Parameters
    // -------------------------------------------------------------------------
    // PIXEL_W: Width of each fixed-point data word in Q7.10 signed format.
    //          18 bits = 1 sign + 7 integer + 10 fractional.
    //          Used for pixel inputs, weight values, and MAC outputs.
    parameter int PIXEL_W        = 18,

    // MAC_OUT_W: Width of each conv9 accumulator output.
    //            40 bits = 9 DSP48E2 cascade outputs (each adds 18x18=36-bit
    //            product, accumulated to 40 bits). Upper bits discarded later.
    parameter int MAC_OUT_W      = 40,

    // DATA_WIDTH: Internal data width for adder tree and Shaaban units.
    //             Matches PIXEL_W = 18 bits.
    parameter int DATA_WIDTH     = 18,

    // N_SHAABAN: Number of Shaaban processing units.
    //            32 units are always instantiated. Stage 1 uses all 32,
    //            Stage 2 uses 3, Stage 3 uses 1.
    parameter int N_SHAABAN      = 32,

    // INPUTS_PER_SHB: Number of conv outputs fed into each Shaaban unit.
    //                 4 inputs = one 2x2 MaxPool window worth of CONV25 results.
    parameter int INPUTS_PER_SHB = 4,

    // FRAME_NO: Number of spatial frames used to sweep the input grid.
    //           6 frames cover the full spatial area of the input feature map.
    //           See README Section 8.3 for the 6-frame spatial strategy.
    parameter int FRAME_NO       = 6,

    // FRAME_NO_WIDTH: Bit width of the frame selector signal.
    //                 Must be ceil(log2(6)) = 3 to represent values 1-6.
    //                 Written as a literal (3) because Vivado cannot evaluate
    //                 $clog2(FRAME_NO) as a parameter default during elaboration.
    parameter int FRAME_NO_WIDTH = 3
)(
    // -------------------------------------------------------------------------
    // Global Control
    // -------------------------------------------------------------------------
    input  logic                          clk,      // 40 MHz system clock
    input  logic                          rst,      // Synchronous active-high reset
    input  wire                           arst_n,   // Async active-low reset
    //                                              // (used by mem_mapping module)
    input  wire                           enable,   // Global enable

    // -------------------------------------------------------------------------
    // Stage 1 Pixel Memory Bus
    // -------------------------------------------------------------------------
    // 6912-bit flat bus (384 pixels x 18 bits each = 6912 bits).
    // Carries raw grayscale pixel values in 18-bit Q7.10 signed format.
    // Sourced from the raw pixel BRAM read port.
    // This bus is used ONLY during Stage 1 (src_sel=2'b00).
    // It is NOT connected to spike_mem -- they are two separate BRAMs.
    //
    // Packing convention: pixel i occupies bits [i*18+17 : i*18] (little-endian).
    // The gen_inmem generate block unpacks this into in_mem[0..383].
    //
    // TODO: When BRAM is instantiated, connect the BRAM read data port here.
    input  wire  [6911:0]                 pixel_mem,

    // -------------------------------------------------------------------------
    // Stage 2/3 Spike Memory Bus
    // -------------------------------------------------------------------------
    // 3200-bit flat bus (3200 x 1-bit spike values).
    // Carries binary spike outputs from Stage 1, written to memory by
    // mem_maping_1_2 at the end of Stage 1 processing.
    // Used during Stage 2/3 (src_sel=2'b01 or 2'b10).
    // It is NOT connected to pixel_mem -- they are two separate BRAMs.
    //
    // Why 3200 bits? Stage 1 produces 32 Shaabans x 100 spatial positions = 3200
    // spike values, packed 1-bit each into a flat 3200-bit word.
    // The mem_mapping module reads specific bits of this bus based on frame.
    // During Stage 3, the first 1024 bits carry the Stage 2 writeback layout:
    // 64 channels x 16 positions = 4x4x64. bin_muxing_stage2 converts that
    // compact layout into four 3x3 windows for the Stage 3 conv path.
    //
    // TODO: When BRAM is instantiated, connect the BRAM read data port here.
    // The same BRAM that mem_maping_1_2 writes must be read here for Stage 2.
    input  wire  [3199:0]                 spike_mem,

    // -------------------------------------------------------------------------
    // Shaaban Hyperparameters
    // -------------------------------------------------------------------------
    // These three values are shared across all 32 Shaaban units.
    // Per-channel variation is folded into the weight ROM values during the
    // training export step -- these are global scalars only.
    //
    // conv_bias:    additive bias applied in conv_bias_Relu sub-module
    // mult_weight:  multiplicative scale in Batch_Norm (Batch_Norm.v line: out = in*mult)
    // add_weight:   additive shift in Batch_Norm (Batch_Norm.v line: out += add)
    input  logic signed [DATA_WIDTH-1:0]  conv_bias,
    input  logic signed [DATA_WIDTH-1:0]  mult_weight,
    input  logic signed [DATA_WIDTH-1:0]  add_weight,

    // -------------------------------------------------------------------------
    // Spike Outputs
    // -------------------------------------------------------------------------
    // 32 binary spike outputs, one per Shaaban unit.
    // During Stage 1: all 32 bits carry valid spike data.
    // During Stage 2: only bits [2:0] carry valid data (units 0,1,2 active).
    // During Stage 3: only bit [0] carries valid data (unit 0 active).
    // Inactive bits are driven to 0 by the adder tree MUX (zero input to LIF).
    //
    // These spikes also feed mem_maping_1_2 for write-back.
    // They will eventually feed the classifier head (GAP -> FC1 -> FC2).
    // TODO: connect to classifier head when implemented.
    output logic [N_SHAABAN-1:0]          spike_out,

    // -------------------------------------------------------------------------
    // Controller Status
    // -------------------------------------------------------------------------
    output logic                          done

    // NOTE: mem_mapped is intentionally NOT a top-level output port.
    // If declared as "output logic [0:31] mem_mapped [0:3199]", it would
    // create 3200 x 32 = 102,400 physical I/O pins on the FPGA, which is
    // impossible (the XCVU11P has only ~2500 user I/O pins in this package).
    // Instead, mem_maping_1_2 output is wired internally to the BRAM write port.
    // BRAM control signals are generated by the internal top_controller and
    // will connect directly when the spike BRAM is instantiated.
);

    // =========================================================================
    // INTERNAL SIGNAL DECLARATIONS
    // =========================================================================
    // Signals are grouped by which pipeline block they connect.
    // Naming convention:
    //   *_s1      : Stage 1 specific (before src_sel MUX)
    //   *_s2      : Stage 2/3 specific (before src_sel MUX)
    //   *_mapped  : after src_sel MUX, feeds shared hardware
    //   mac_*     : conv9 array outputs
    //   shb_*     : Shaaban unit inputs/outputs
    // =========================================================================

    // -------------------------------------------------------------------------
    // Controller outputs
    // -------------------------------------------------------------------------

    // top_controller owns every stage-control signal used by this datapath.
    // The BRAM-related controls are declared now and will connect directly to
    // the spike memory when that memory is instantiated in this module.
    logic [1:0]                    src_sel;
    logic [FRAME_NO_WIDTH-1:0]     frame;
    logic                          stage_sel;
    logic [5:0]                    conv2_filter;
    logic [6:0]                    conv3_filter;
    logic [0:3199]                 ctrl_mem_enable;
    logic                          ctrl_rd_enable;
    logic [5:0]                    ctrl_rd_mem_adderss;
    logic [5:0]                    ctrl_wr_mem_adderss;
    logic                          ctrl_zero;
    logic                          ctrl_zero_sel;
    logic                          ctrl_padding_flag;

    top_controller u_top_controller (
        .clk            (clk),
        .rst            (rst),
        .arst_n         (arst_n),
        .enable         (enable),
        .mem_enable     (ctrl_mem_enable),
        .rd_enable      (ctrl_rd_enable),
        .stage          (src_sel),
        .frame          (frame),
        .stage_sel      (stage_sel),
        .conv2_filter   (conv2_filter),
        .conv3_filter   (conv3_filter),
        .rd_mem_adderss (ctrl_rd_mem_adderss),
        .wr_mem_adderss (ctrl_wr_mem_adderss),
        .zero           (ctrl_zero),
        .zero_sel       (ctrl_zero_sel),
        .padding_flag   (ctrl_padding_flag),
        .done           (done)
    );

    // -------------------------------------------------------------------------
    // PATH A: Stage 1 signals
    // -------------------------------------------------------------------------

    // in_mem: 384 signed 18-bit pixel words, unpacked from pixel_mem port.
    // Each word is one pixel from the 24x24 input crop, in Q7.10 format.
    // Indexing: in_mem[block * 32 + filter] for block=0..11, filter=0..31.
    logic signed [PIXEL_W-1:0] in_mem [0:383];

    // p_imag: shift-corrected version of in_mem, produced by Stage1_in logic.
    // Same shape and format as in_mem, but pixels are re-ordered to solve the
    // orphan alignment problem (see Stage1_in generate blocks below).
    logic signed [PIXEL_W-1:0] p_imag [0:383];

    // pixels_s1: final Stage 1 pixel input array for the conv9 instances.
    // Shape [12 blocks][32 filters][9 taps].
    // All 9 taps of each 3x3 window get the same p_imag value -- spatial
    // differentiation across taps is handled entirely by the weight ROM values.
    logic signed [PIXEL_W-1:0] pixels_s1 [0:11][0:31][0:8];

    // -------------------------------------------------------------------------
    // PATH B: Stage 2/3 signals
    // -------------------------------------------------------------------------

    // fil_in: output of mem_mapping module.
    // 32 filters x 40 spike positions, 1-bit each.
    // For each filter and position, the correct spike bit is selected from
    // spike_mem based on the current frame value.
    logic fil_in [31:0][39:0];

    // conv_windows: output of the 32 frame_input_mapping instances.
    // [filter][engine][tap] -- 32 x 12 x 9 = 3456 1-bit values.
    // Represents 12 overlapping 3x3 windows per filter for all 32 filters.
    logic conv_windows [31:0][11:0][8:0];

    // pixels_s2: Stage 2 pixel input array after sign-extension.
    // Shape [12][32][9], same as pixels_s1 but sourced from conv_windows.
    // 1-bit spikes sign-extended to 18-bit: 0 --> 18'h00000, 1 --> 18'h3FFFF.
    logic signed [PIXEL_W-1:0] pixels_s2 [0:11][0:31][0:8];

    // stage3_mem: unpacked view of the first 1024 spike_mem bits.
    // Stage 2 writeback stores a 4x4 map for each of 64 channels:
    //   stage3_mem[channel*16 + spatial_index].
    logic stage3_mem [0:1023];

    // stage3_windows: output of bin_muxing_stage2.
    // Shape [tap][window][channel] = 9 x 4 x 64.
    // The four windows are the four 3x3 conv windows inside the 4x4 Stage 3 input.
    logic stage3_windows [0:8][0:3][0:63];

    // pixels_s3: Stage 3 pixel input array after sign-extension.
    // Rows 0/1, 2/3, 4/5, 6/7 hold the four Stage 3 windows split across
    // channels 0..31 and 32..63. Rows 8..11 are tied to zero.
    logic signed [PIXEL_W-1:0] pixels_s3 [0:11][0:31][0:8];

    // -------------------------------------------------------------------------
    // SHARED: Active pixel array (MUX output, feeds conv9 array)
    // -------------------------------------------------------------------------

    // pixels_mapped: the actual pixel inputs seen by the conv9 array.
    // src_sel MUX chooses between pixels_s1, pixels_s2, and pixels_s3.
    // Shape [12 blocks][32 filters][9 taps].
    logic signed [PIXEL_W-1:0] pixels_mapped [0:11][0:31][0:8];

    // -------------------------------------------------------------------------
    // Weight ROM outputs
    // -------------------------------------------------------------------------

    // Three separate weight arrays, one per stage.
    // All stored in distributed LUTs (not BRAMs) so every conv9 unit can
    // read its own weight simultaneously -- unlimited parallel read ports,
    // zero read latency, no arbitration needed.
    // Each array has 3456 entries: 384 conv9 units x 9 taps each.
    logic [PIXEL_W-1:0] stage1_weights [3456];   // 5x5 kernels, decomposed
    logic [PIXEL_W-1:0] stage2_weights [3456];   // 3x3 kernels, 32-channel
    logic [PIXEL_W-1:0] stage3_weights [3456];   // 3x3 kernels, 64-channel

    // active_weights: the weight ROM selected by src_sel MUX.
    // This feeds the weights_mapped unpacking below.
    logic [PIXEL_W-1:0] active_weights [3456];

    // weights_mapped: unpacked weight array for conv9 instances.
    // Shape [12][32][9] matching pixels_mapped, sourced from active_weights.
    // Index formula: active_weights[(block * 32 + filter) * 9 + tap]
    logic signed [PIXEL_W-1:0] weights_mapped [0:11][0:31][0:8];

    // -------------------------------------------------------------------------
    // conv9 array outputs
    // -------------------------------------------------------------------------

    // mac_raw: raw 40-bit accumulator outputs from each conv9 unit.
    // Shape [12 blocks][32 filters].
    // Each value = sum of 9 signed 18-bit multiply-accumulate operations.
    logic [MAC_OUT_W-1:0] mac_raw [0:11][0:31];

    // mac_to_connect: mac_raw truncated to DATA_WIDTH=18 bits.
    // This is what the adder tree receives as input.
    // IMPORTANT: Declared as UNPACKED array [0:11][0:31] (not packed multi-dim)
    // to match the port declaration of adder_tree_shaaban_connect exactly.
    // Using a packed declaration [11:0][31:0][17:0] would reverse the row/col
    // indexing in Vivado elaboration and cause undriven DSP input warnings.
    logic signed [DATA_WIDTH-1:0] mac_to_connect [0:11][0:31];

    // -------------------------------------------------------------------------
    // Shaaban unit bus
    // -------------------------------------------------------------------------

    // shb_bus: packed input bus for each Shaaban unit.
    // Each entry holds INPUTS_PER_SHB=4 packed DATA_WIDTH=18-bit values.
    // Total width per entry: 4 x 18 = 72 bits.
    // Sourced from the adder_tree_shaaban_connect module's src_sel MUX output.
    logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_bus [0:N_SHAABAN-1];

    // -------------------------------------------------------------------------
    // Write-back signals (internal only -- NOT top-level ports)
    // -------------------------------------------------------------------------

    // shaaban_spike_bus: Shaaban spike outputs formatted for mem_maping_1_2.
    // mem_maping_1_2 expects shaaban_out[0:31] each as logic [0:31] (32-bit word).
    // Each 1-bit spike is replicated 32 times to fill its 32-bit word slot.
    logic [0:31] shaaban_spike_bus [0:31];

    // mem_mapped_internal: output of mem_maping_1_2.
    // 3200 x 32-bit words representing the spike layout for the next stage's BRAM.
    // This is INTERNAL and does NOT appear as a top-level port (would need 102,400 pins).
    // TODO: Connect to BRAM write data port when BRAM is instantiated below.
    logic [0:31] mem_mapped_internal [0:3199];

    // =========================================================================
    // PATH A: STAGE 1 INPUT PIPELINE
    // =========================================================================

    // -------------------------------------------------------------------------
    // A1: Unpack pixel_mem into in_mem array
    // -------------------------------------------------------------------------
    // pixel_mem is a flat 6912-bit bus (384 x 18-bit words packed together).
    // We slice it into the 384-element in_mem array so Stage1_in can process it.
    //
    // Slicing is safe here because pixel_mem is declared exactly 6912 bits wide.
    // This avoids the "part-select out of range" error that occurred in a previous
    // version when we tried to slice 18-bit words from a 3200-bit bus (impossible
    // because 384*18=6912 >> 3200).
    genvar m;
    generate
        for (m = 0; m < 384; m++) begin : gen_inmem
            assign in_mem[m] = $signed(pixel_mem[m*PIXEL_W +: PIXEL_W]);
        end
    endgenerate

    // -------------------------------------------------------------------------
    // A2: Stage1_in shift-mapping  (in_mem[384] --> p_imag[384])
    // -------------------------------------------------------------------------
    // This implements the cyclic 0/1/2-shift strategy described in README
    // Section 5.2.1 "Data Routing and Input Mapping for CONV25".
    //
    // THE PROBLEM IT SOLVES:
    //   The adder tree groups conv9 outputs in triplets for CONV25 formation.
    //   Each block has 32 conv9 outputs. 32 / 3 = 10 complete triplets + 2 leftover.
    //   These 2 leftovers (positions 30 and 31 of each block) cannot form a
    //   complete CONV25 group within their own block.
    //
    // THE SOLUTION:
    //   Pre-route the pixel stream so the 2 leftovers from block B naturally
    //   align with the first element of block B+1, forming a cross-block triplet.
    //   The pattern repeats every 3 blocks (since 3*32=96, divisible by 3):
    //
    //     Block % 3 == 0  (Blocks 0,3,6,9):  straight copy, no offset
    //     Block % 3 == 1  (Blocks 1,4,7,10): 1-position shift forward
    //     Block % 3 == 2  (Blocks 2,5,8,11): 2-position shift forward
    //
    // IMPLEMENTATION NOTE ON THREE SEPARATE LOOPS:
    //   We use three separate generate-for loops with independent if() conditions
    //   rather than a single loop with if/else if/else. This is because some
    //   EDA tools (including older Vivado versions) do not handle generate-if
    //   else-chains consistently inside generate-for loops. Three separate loops
    //   produce identical hardware and are universally supported. Each p_imag[i]
    //   index satisfies exactly one modulo condition, so there are no multiple-
    //   driver or undriven-signal issues.

    genvar b, j;
    generate

        // ---
        // State 0: Straight copy (Blocks 0, 3, 6, 9 where b % 3 == 0)
        // ---
        // Block is aligned -- in_mem maps directly to p_imag with no offset.
        // Positions 0..29 feed 10 complete triplets (10 x 3 = 30 conv9 inputs).
        // Positions 30..31 are "orphans" -- they will be consumed by the
        // cross-block grouping handled by the ext_sum_correction in the adder tree.
        for (b = 0; b < 12; b = b + 1) begin : gen_state_0
            if (b % 3 == 0) begin : state_0
                for (j = 0; j < 32; j = j + 1) begin : assign_s0
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j];
                end
            end
        end

        // ---
        // State 1: 1-position shift (Blocks 1, 4, 7, 10 where b % 3 == 1)
        // ---
        // Positions 0..29: take in_mem[pos+1] (shifted forward by 1 slot).
        //   This shifts the 30 main inputs, leaving room for cross-block grouping.
        // Position 30: wraps back to in_mem[block_start] (the skipped element).
        //   This element pairs with the 2 orphans from the previous block
        //   to form one of the 8 correction groups in ext_sum_correction.
        // Position 31: takes in_mem[block_end] as a seed for the next block's orphan.
        //   This becomes the 1 remaining orphan from this block.
        for (b = 0; b < 12; b = b + 1) begin : gen_state_1
            if (b % 3 == 1) begin : state_1
                for (j = 0; j < 30; j = j + 1) begin : assign_s1
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 1];
                end
                assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];       // wrap: skipped element
                assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 31];  // seed: 1 orphan remains
            end
        end

        // ---
        // State 2: 2-position shift (Blocks 2, 5, 8, 11 where b % 3 == 2)
        // ---
        // Positions 0..29: take in_mem[pos+2] (shifted forward by 2 slots).
        // Position 30: wraps to in_mem[block_start+0] (first skipped element).
        // Position 31: wraps to in_mem[block_start+1] (second skipped element).
        //   The two wrap-arounds fully consume the 1 orphan from the previous block.
        //   After this block, zero orphans remain -- alignment resets to State 0.
        for (b = 0; b < 12; b = b + 1) begin : gen_state_2
            if (b % 3 == 2) begin : state_2
                for (j = 0; j < 30; j = j + 1) begin : assign_s2
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 2];
                end
                assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];       // wrap: skipped[0]
                assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 1];   // wrap: skipped[1]
            end
        end

    endgenerate

    // -------------------------------------------------------------------------
    // A3: Build pixels_s1 from p_imag (broadcast to 9 taps)
    // -------------------------------------------------------------------------
    // The conv9 array needs a [12][32][9] input array.
    // In Stage 1, all 9 taps of a given conv9 unit receive the SAME pixel value.
    // The 9 different filter weights (Q[0]..Q[8]) provide the 5x5 kernel spatial
    // differentiation -- different weights applied to the same base pixel create
    // the equivalent of reading 9 different spatial positions.
    //
    // Index mapping: pixels_s1[block][filter][tap] = p_imag[block*32 + filter]
    //                for all tap in 0..8
    genvar gp1, cp1, tp1;
    generate
        for (gp1 = 0; gp1 < 12; gp1++) begin : gen_ps1_row
            for (cp1 = 0; cp1 < 32; cp1++) begin : gen_ps1_col
                for (tp1 = 0; tp1 < 9; tp1++) begin : gen_ps1_tap
                    assign pixels_s1[gp1][cp1][tp1] = p_imag[(gp1 * 32) + cp1];
                end
            end
        end
    endgenerate

    // =========================================================================
    // PATH B: STAGE 2/3 INPUT PIPELINE
    // =========================================================================

    // -------------------------------------------------------------------------
    // B1: Stage 2 mem_mapping (frame_mapping_iterations_filters.sv)
    // -------------------------------------------------------------------------
    // This module reads specific bits from the 3200-bit spike_mem bus and
    // routes them into the fil_in[32][40] output array for Stage 2.
    //
    // WHY THIS IS NEEDED:
    //   Stage 2 processes 32 input channels with 12 parallel conv engines.
    //   The engines need 9 different spike values each (for the 3x3 kernel),
    //   meaning 12 x 9 = 108 reads per cycle. A standard BRAM only has 2 ports.
    //   Solution: pre-load a 3200-bit "strip" of spikes into a flat bus.
    //   Then use pure combinational MUX logic (the case statements inside
    //   mem_mapping) to "point" each engine input at the correct bit position.
    //   Changing the frame signal instantly re-routes all connections -- no data moves.
    //
    // The 1280 case statements inside mem_mapping were generated by a Python
    // "spatial compiler" script (README Section 8.5). Each case implements the
    // linearization formula: Bit_Index = (row x 320) + (col x 32) + filter_offset
    //
    // frame values 1-6 correspond to the 6 spatial frame patterns defined in
    // README Section 8.3. Each frame covers a different region of the grid.
    mem_mapping #(
        .FRAME_NO       (FRAME_NO),
        .FRAME_NO_WIDTH (FRAME_NO_WIDTH),
        .MEM_WORD       (3200)
    ) u_mem_mapping (
        .clk    (clk),
        .arst_n (arst_n),
        .frame  (frame),
        .mem    (spike_mem),   // IMPORTANT: Stage 2 spike bus, NOT pixel_mem
        .fil_in (fil_in)
    );

    // -------------------------------------------------------------------------
    // B2: frame_input_mapping x32 (frame_input_mapping_brackets.sv)
    // -------------------------------------------------------------------------
    // 32 parallel instances, one per output filter channel.
    // Each instance takes fil_in[fi][0:39] (40 1-bit spikes for filter fi)
    // plus the frame signal, and assembles conv_windows[fi][12][9]:
    //   12 overlapping 3x3 windows, each with 9 1-bit tap values.
    //
    // The anchor+offset rules inside are derived from backtracking analysis
    // (README Section 3 and 8.4) and are hard-coded as case statements.
    //
    // This is pure combinational logic -- no registers, no clock needed.
    // The frame[2:0] explicit slice is needed because frame_input_mapping
    // declares its port as input [2:0] frame.
    genvar fi;
    generate
        for (fi = 0; fi < 32; fi++) begin : gen_frame_mapping
            frame_input_mapping u_frame_map (
                .frame (frame[2:0]),
                .in    (fil_in[fi]),
                .conv  (conv_windows[fi])
            );
        end
    endgenerate

    // -------------------------------------------------------------------------
    // B3: Build pixels_s2 from conv_windows (sign-extend 1-bit to 18-bit)
    // -------------------------------------------------------------------------
    // conv_windows values are 1-bit binary spikes (0 or 1).
    // The conv9 DSP inputs require 18-bit signed values.
    // Sign extension maps:
    //   0 --> 18'h00000  (=  0 in Q7.10, inactive spike)
    //   1 --> 18'h3FFFF  (= -1 in Q7.10 signed, active spike)
    //
    // Dimension transpose: conv_windows[filter][engine][tap] is indexed with
    // filter as the outer dimension, but pixels_s2[engine][filter][tap] needs
    // engine as the outer dimension (to match how conv9 instances are indexed
    // as [row=engine][col=filter] in the generate loop below).
    genvar gp2, cp2, tp2;
    generate
        for (gp2 = 0; gp2 < 12; gp2++) begin : gen_ps2_row
            for (cp2 = 0; cp2 < 32; cp2++) begin : gen_ps2_col
                for (tp2 = 0; tp2 < 9; tp2++) begin : gen_ps2_tap
                    assign pixels_s2[gp2][cp2][tp2] =
                        {{(PIXEL_W-1){conv_windows[cp2][gp2][tp2]}},  // sign replicate
                                      conv_windows[cp2][gp2][tp2]};   // LSB
                end
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // B4: Stage 3 bin muxing from Stage 2 writeback layout
    // -------------------------------------------------------------------------
    // Stage 2 produces a compact 4x4 map for each of 64 channels:
    //   64 channels x 16 spatial positions = 1024 bits.
    //
    // bin_muxing_stage2 carves each 4x4 channel map into the four 3x3 windows
    // needed by Stage 3 convolution:
    //   window 0: top-left     window 1: top-right
    //   window 2: bottom-left  window 3: bottom-right
    //
    // These four windows are mapped into the shared 12x32 conv array as
    // adjacent row pairs. The existing adder_tree_shaaban_connect Stage 3 path
    // pairwise-adds tree_final[0]+[1], [2]+[3], [4]+[5], and [6]+[7], giving
    // four full 64-channel conv sums for Shaaban unit 0's 2x2 max-pool input.
    genvar sm;
    generate
        for (sm = 0; sm < 1024; sm++) begin : gen_stage3_mem_unpack
            assign stage3_mem[sm] = spike_mem[sm];
        end
    endgenerate

    bin_muxing_stage2 u_stage3_bin_mux (
        .din  (stage3_mem),
        .dout (stage3_windows)
    );

    genvar win3, ch3, tap3;
    generate
        for (win3 = 0; win3 < 4; win3++) begin : gen_ps3_window
            for (ch3 = 0; ch3 < 32; ch3++) begin : gen_ps3_channel
                for (tap3 = 0; tap3 < 9; tap3++) begin : gen_ps3_tap
                    assign pixels_s3[(win3 * 2)    ][ch3][tap3] =
                        {{(PIXEL_W-1){stage3_windows[tap3][win3][ch3]}},
                                      stage3_windows[tap3][win3][ch3]};

                    assign pixels_s3[(win3 * 2) + 1][ch3][tap3] =
                        {{(PIXEL_W-1){stage3_windows[tap3][win3][ch3 + 32]}},
                                      stage3_windows[tap3][win3][ch3 + 32]};
                end
            end
        end
    endgenerate

    genvar zrow3, zch3, ztap3;
    generate
        for (zrow3 = 8; zrow3 < 12; zrow3++) begin : gen_ps3_zero_row
            for (zch3 = 0; zch3 < 32; zch3++) begin : gen_ps3_zero_channel
                for (ztap3 = 0; ztap3 < 9; ztap3++) begin : gen_ps3_zero_tap
                    assign pixels_s3[zrow3][zch3][ztap3] = '0;
                end
            end
        end
    endgenerate

    // =========================================================================
    // SHARED: src_sel PIXEL MUX
    // =========================================================================
    // Selects which input path feeds the conv9 array.
    // This is a pure combinational 2-to-1 MUX replicated across all 12x32x9
    // pixel positions. Zero additional pipeline delay is introduced.
    //
    // Stage-specific paths:
    //   Stage 1 reads raw pixel data.
    //   Stage 2 reads the 3200-bit Stage 1 writeback layout through frame logic.
    //   Stage 3 reads the 1024-bit Stage 2 writeback layout through bin_muxing_stage2.
    genvar gm, cm, tm;
    generate
        for (gm = 0; gm < 12; gm++) begin : gen_pmux_row
            for (cm = 0; cm < 32; cm++) begin : gen_pmux_col
                for (tm = 0; tm < 9; tm++) begin : gen_pmux_tap
                    always_comb begin
                        case (src_sel)
                            2'b00:   pixels_mapped[gm][cm][tm] = pixels_s1[gm][cm][tm];  // Stage 1
                            2'b01:   pixels_mapped[gm][cm][tm] = pixels_s2[gm][cm][tm];  // Stage 2
                            2'b10:   pixels_mapped[gm][cm][tm] = pixels_s3[gm][cm][tm];  // Stage 3
                            default: pixels_mapped[gm][cm][tm] = pixels_s1[gm][cm][tm];
                        endcase
                    end
                end
            end
        end
    endgenerate

    // =========================================================================
    // WEIGHT ROMs (Distributed LUT storage)
    // =========================================================================
    // All weight values are stored in Xilinx distributed LUTs, NOT in BRAMs.
    //
    // WHY DISTRIBUTED LUTs, NOT BRAMs?
    //   BRAMs have a limited number of read ports (typically 2 per RAMB36E2).
    //   The conv9 array needs to read 3456 different weight values simultaneously
    //   (one per DSP per cycle). Distributing weights across LUTs gives effectively
    //   unlimited parallel read ports with zero clock cycles of latency.
    //
    // All three weight ROMs are instantiated and powered on at all times.
    // The src_sel MUX below selects which ROM's output reaches active_weights.
    // The "losing" ROMs drive their outputs into unconnected active_weights paths,
    // which Vivado optimises away -- no extra power or area cost.
    //
    // Weight ROM flat array size: 384 conv9 units x 9 taps = 3456 entries.
    // Each entry is PIXEL_W=18 bits wide.
    // Index formula: entry for block g, filter c, tap t = (g*32 + c)*9 + t

    // Stage 1 weights: 5x5 kernel decomposed into 3x conv9 (27 weights, 2 zeros)
    // Source module: CONV1_W_MAP_OPT.sv (package-based constant assignments)
    CONV1_W_MAP_OPT u_w1 (
        .conv9_in (stage1_weights)
    );

    // Stage 2 weights: 3x3 kernel, 32 input channels
    // Source module: CONV2_W_MAP_OPT.sv
    CONV2_W_MAP_OPT u_w2 (
        .filter   (conv2_filter),
        .conv9_in (stage2_weights)
    );

    // Stage 3 weights: 3x3 kernel, 64 input channels
    // Source module: CONV3_W_MAP_OPT.sv
    CONV3_W_MAP_OPT u_w3 (
        .filter   (conv3_filter),
        .conv9_in (stage3_weights)
    );

    // src_sel MUX: select active weight ROM
    always_comb begin
        case (src_sel)
            2'b00:   active_weights = stage1_weights;   // Stage 1
            2'b01:   active_weights = stage2_weights;   // Stage 2
            2'b10:   active_weights = stage3_weights;   // Stage 3
            default: active_weights = stage2_weights;   // safe default
        endcase
    end

    // -------------------------------------------------------------------------
    // Weight array unpacking: flat active_weights[3456] --> shaped [12][32][9]
    // -------------------------------------------------------------------------
    // The conv9 instances need weights indexed as weights_mapped[block][filter][tap].
    // We unpack the flat active_weights using the index formula above.
    genvar gw, cw, tw;
    generate
        for (gw = 0; gw < 12; gw++) begin : gen_wmap_row
            for (cw = 0; cw < 32; cw++) begin : gen_wmap_col
                for (tw = 0; tw < 9; tw++) begin : gen_wmap_tap
                    assign weights_mapped[gw][cw][tw] =
                        $signed(active_weights[(gw * 32 + cw) * 9 + tw]);
                end
            end
        end
    endgenerate

    // =========================================================================
    // CONV9 ARRAY: 12 x 32 = 384 parallel conv9 units
    // =========================================================================
    // This is the central computational block of the accelerator.
    // 384 conv9 units run in parallel every clock cycle.
    //
    // Each conv9 computes:
    //   Pixel_Out = P[0]*Q[0] + P[1]*Q[1] + ... + P[8]*Q[8]
    // using a cascade of 9 DSP48E2 slices (one multiply-accumulate per DSP).
    // Output is a 40-bit signed accumulator value.
    //
    // The same 384 conv9 units serve ALL three stages. The src_sel MUX above
    // routes different pixel sources and weight ROMs to the same DSP fabric.
    //
    // Physical layout: row g = block index (0..11), col c = filter index (0..31).
    // Each (g,c) pair is one conv9 instance handling one spatial position and
    // one output filter channel.
    genvar g, c;
    generate
        for (g = 0; g < 12; g++) begin : gen_conv_row
            for (c = 0; c < 32; c++) begin : gen_conv_col
                conv9 #(
                    .PIXEL_W (PIXEL_W),   // 18-bit Q7.10 signed inputs
                    .PROD_W  (36),        // 18x18 product width
                    .OUT_W   (MAC_OUT_W)  // 40-bit accumulator
                ) u_conv (
                    .CLK       (clk),
                    .P         (pixels_mapped[g][c]),   // 9 pixel taps
                    .Q         (weights_mapped[g][c]),  // 9 weight taps
                    .Pixel_Out (mac_raw[g][c])          // 40-bit MAC result
                );
            end
        end
    endgenerate

    // -------------------------------------------------------------------------
    // Truncation: 40-bit MAC --> 18-bit for adder tree
    // -------------------------------------------------------------------------
    // We keep only the lower DATA_WIDTH=18 bits of each 40-bit MAC result.
    // The upper 22 bits are discarded.
    //
    // Why is this safe?
    //   The weight values and pixel values are constrained during training such
    //   that the 18-bit range is sufficient to represent the convolution output.
    //   Overflow protection is enforced by the quantisation constraints applied
    //   during the weight export step. The upper bits will be zero or sign-extended
    //   copies of bit 17 for valid results.
    //
    // mac_to_connect is UNPACKED [0:11][0:31] (not packed multi-dimensional).
    // This avoids the synthesis error where Vivado reversed row/col indexing
    // when the signal was declared as packed [11:0][31:0][17:0].
    genvar g2, c2;
    generate
        for (g2 = 0; g2 < 12; g2++) begin : gen_trunc_row
            for (c2 = 0; c2 < 32; c2++) begin : gen_trunc_col
                assign mac_to_connect[g2][c2] = mac_raw[g2][c2][DATA_WIDTH-1:0];
            end
        end
    endgenerate

    // =========================================================================
    // ADDER TREE + STAGE ROUTING: adder_tree_shaaban_connect
    // =========================================================================
    // This module performs the summation and routes results to the correct
    // Shaaban units based on the active stage.
    //
    // INTERNALLY, this module contains (see adder_tree_shaaban_connect.sv):
    //
    //   1. 12 x adder_tree_10_4_1_1 (one per block row g=0..11)
    //      Each tree receives mac_in[g][0:31] (32 conv9 outputs from one row).
    //      It produces:
    //        - tree_tap[0..9]: 10 Layer-1 partial sums, each summing 3 conv9
    //                          outputs --> these are the CONV25 results
    //                          (3 conv9 in parallel = 27 MACs, covers 25 of 27
    //                          with 2 zero weights, see README Section 4.4)
    //        - final_output: full sum of all 32 conv9 outputs in this row
    //                         --> used for Stage 2/3 path
    //
    //   2. ext_sum_correction (orphan correction layer)
    //      The 2 orphan outputs from each of the 12 trees (slots 30 and 31)
    //      cannot form complete CONV25 groups within their block.
    //      ext_sum_correction collects all 24 orphan values (12 x 2)
    //      and groups them into 8 three-input correction sums (corr_out[0..7]).
    //      These 8 values complete the 128-input flat_s1 array.
    //
    //   3. flat_s1[128] assembly
    //      128 = 32 Shaabans x 4 inputs each = total inputs needed for Stage 1.
    //      Assembled from 120 tree taps + 8 correction outputs:
    //        Trees 0-7  (8 trees): each contributes 11 values (10 taps + 1 correction)
    //        Trees 8-11 (4 trees): each contributes 10 values (10 taps, no correction)
    //        Total: 8x11 + 4x10 = 88 + 40 = 128 values
    //
    //   4. 3-way src_sel MUX (same src_sel as top-level):
    //        2'b00 Stage 1: shb_bus[s] = flat_s1[s*4 .. s*4+3]  for s=0..31
    //                       All 32 Shaaban units receive 4 CONV25 values each.
    //        2'b01 Stage 2: shb_bus[0] = {tree_final[0..3]}
    //                       shb_bus[1] = {tree_final[4..7]}
    //                       shb_bus[2] = {tree_final[8..11]}
    //                       shb_bus[3..31] = 0 (inactive)
    //                       Only 3 Shaabans receive valid data.
    //        2'b10 Stage 3: shb_bus[0] = {s3_results[0..3]}
    //                       Each s3_result is one 64-channel accumulated window.
    //                       All other units are zero. Only 1 Shaaban receives valid data.
    //
    // KEY POINT: The adder tree hardware itself does not change between stages.
    // The same 12 trees always compute all their outputs (taps AND final sum)
    // every cycle. The src_sel MUX simply routes different outputs to Shaabans.
    // No extra hardware is added for stage switching -- this is the resource-
    // sharing innovation described in README Section 1.4.
    adder_tree_shaaban_connect u_connect (
        .clk          (clk),
        .rst          (rst),
        .src_sel      (src_sel),        // stage selector -- routes MUX inside
        .mac_in       (mac_to_connect), // 12x32 truncated conv9 outputs
        .shb_conv_bus (shb_bus)         // 32 packed Shaaban input buses
    );

    // =========================================================================
    // SHAABAN PROCESSING ARRAY: 32 units, always instantiated
    // =========================================================================
    // All 32 Shaaban units (shaban_unit_top) are permanently on the FPGA.
    // Units that are not active for the current stage receive conv_in=0 from
    // the adder tree MUX. With zero input they consume no dynamic power and
    // their LIF membrane potential does not change.
    //
    // INTERNAL PIPELINE PER UNIT (verified from RTL, 13 DSPs each):
    //   conv_in (4 x 18-bit packed)
    //       |
    //       +-- [0] --> conv_bias_Relu --> Batch_Norm --> \
    //       +-- [1] --> conv_bias_Relu --> Batch_Norm -->  MaxPool(2:1) --> \
    //       +-- [2] --> conv_bias_Relu --> Batch_Norm -->  MaxPool(2:1) -->  MaxPool(2:1) --> LIF --> spike
    //       +-- [3] --> conv_bias_Relu --> Batch_Norm --> /
    //
    //   conv_bias_Relu: adds conv_bias, then ReLU (clips negatives to 0)
    //   Batch_Norm:     out = (in * mult_weight) + add_weight, take bits [36:19]
    //   MaxPool(2:1):   out = max(in_a, in_b), reduces 4 inputs to 2 then 1
    //   LIF:            Leaky Integrate-and-Fire neuron
    //                   mem_decay = mem_reg >>> 1  (beta=0.5 decay)
    //                   mem_new   = mem_decay + pool_out
    //                   spike     = (mem_new >= threshold) ? 1 : 0
    //                   threshold = 18'd5
    //
    // Active units by stage:
    //   Stage 1 (src_sel=2'b00): units 0-31  (all 32 active)
    //   Stage 2 (src_sel=2'b01): units 0,1,2 (3 active, producing 3 LIF2 outputs/cycle)
    //   Stage 3 (src_sel=2'b10): unit 0 only (1 active, producing 1 LIF3 output/cycle)
    genvar s;
    generate
        for (s = 0; s < N_SHAABAN; s++) begin : gen_shaaban_array
            shaban_unit_top #(
                .DATA_WIDTH         (DATA_WIDTH),        // 18-bit data
                .conv_bias_relu_num (INPUTS_PER_SHB),    // 4 bias+relu units
                .batch_norm_num     (INPUTS_PER_SHB),    // 4 batch norm units
                .pool_num           (2)                  // 2:1 max pool stages
            ) u_shb (
                .clk        (clk),
                .rst        (rst),
                .conv_in    (shb_bus[s]),      // 4 packed 18-bit inputs
                .conv_bias  (conv_bias),       // shared bias scalar
                .mult_wight (mult_weight),     // shared BN scale scalar
                .add_wight  (add_weight),      // shared BN shift scalar
                .spike      (spike_out[s])     // 1-bit LIF output
            );

            // Replicate the 1-bit spike into a 32-bit word for mem_maping_1_2.
            // mem_maping_1_2 port: shaaban_out[0:31] each declared as [0:31],
            // expecting 32-bit word per unit. The 32-bit replication means each
            // spike bit fills all 32 positions, which is then filtered by the
            // stage-specific layout patterns inside mem_maping_1_2.
            assign shaaban_spike_bus[s] = {32{spike_out[s]}};
        end
    endgenerate

    // =========================================================================
    // WRITE-BACK: mem_maping_1_2  (INTERNAL -- NOT a top-level port)
    // =========================================================================
    // Converts the 32 Shaaban spike outputs into the 3200-element flat memory
    // layout required as input for the next stage's mem_mapping module.
    //
    // stage_sel = 1'b0 (Stage 1 --> Stage 2 writeback):
    //   32 Shaaban units x 100 spatial cycles = 3200 spike values total.
    //   The module fills mem_mapped[0..3199] using a 100-group x 32-filter pattern.
    //   Each filter's spike at spatial position i goes to mem_mapped[i*32 + filter].
    //
    // stage_sel = 1'b1 (Stage 2 --> Stage 3 writeback):
    //   Only 3 Shaaban units x ~64 spatial cycles produce valid data.
    //   The module uses a 64 x 16-location grouped pattern.
    //   Unused locations are set to zero (natural zero-padding).
    //
    // WHY NOT A TOP-LEVEL PORT?
    //   mem_mapped has type logic [0:31] mem_mapped [0:3199].
    //   If declared as a top-level output, this creates 3200 x 32 = 102,400
    //   physical I/O pins. The XCVU11P has ~2500 user I/O pins in this package.
    //   This would fail placement with [Place 30-415] overutilization error.
    //   Solution: keep it internal and connect directly to the BRAM write port.
    //
    // TODO: Instantiate the spike output BRAM (RAMB36E2 or UltraRAM) here.
    //       Connect mem_mapped_internal to the BRAM write data input.
    //       Add a write address counter that increments each valid output cycle.
    //       Add a write enable signal driven by the controller.
    //       The same BRAM read port must connect to spike_mem input port above.
    mem_maping_1_2 u_writeback (
        .stage_sel   (stage_sel),           // layout selector (0=Stage1, 1=Stage2)
        .shaaban_out (shaaban_spike_bus),    // 32 x 32-bit spike words
        .mem_mapped  (mem_mapped_internal)  // 3200 x 32-bit output (internal only)
    );

    // =========================================================================
    // TODO: BRAM INSTANTIATION (not yet implemented)
    // =========================================================================
    // When the controller is ready, instantiate a RAMB36E2 or UltraRAM here:
    //
    // RAMB36E2 #(...) u_spike_bram (
    //     .CLKARDCLK   (clk),
    //     .ENARDEN     (enable),
    //     // Write port (from mem_maping_1_2):
    //     .CLKBWRCLK   (clk),
    //     .ENBWREN     (bram_write_enable),   // driven by controller
    //     .ADDRBWRADDR (bram_write_addr),     // driven by controller counter
    //     .DINBDIN     (mem_mapped_internal), // from mem_maping_1_2
    //     // Read port (feeds back to spike_mem input):
    //     .ADDRARDADDR (bram_read_addr),      // driven by controller
    //     .DOUTADOUT   (spike_mem_internal),  // connects to spike_mem input
    //     ...
    // );
    //
    // Also need to add BRAM control signals as top-level ports or
    // instantiate the controller module here to drive them internally.
    // =========================================================================

endmodule
