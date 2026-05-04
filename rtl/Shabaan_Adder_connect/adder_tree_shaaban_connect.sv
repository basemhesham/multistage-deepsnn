`timescale 1ns / 1ps
// =============================================================================
// adder_tree_shaaban_connect.sv
// -----------------------------------------------------------------------------
// TOP-LEVEL GLUE: Adder Trees → Shaaban Units
//
// SYSTEM CONTEXT
//   This module is the datapath bridge for Stage 1 of the MultiStage-DeepSNN
//   CNN-SNN hybrid. It sits between:
//     ← 12 adder_tree_10_4_1_1 instances  (receive 384 MAC products)
//     → 32 shaban_unit_top instances       (produce 32 spike outputs)
//
//   The same 32 Shaaban units are REUSED across all three pipeline stages
//   via a 3-way MUX controlled by src_sel. Only the input source changes.
//
// STAGE ROUTING  (src_sel)
//  ┌──────────┬─────────┬──────────────────────────────────────────────────┐
//  │ src_sel  │  Stage  │ Description                                      │
//  ├──────────┼─────────┼──────────────────────────────────────────────────┤
//  │  2'b00   │    1    │ 128 conv25 outputs → all 32 Shaabans (4 each)    │
//  │  2'b01   │    2    │ 12 tree finals → Shaabans 0,1,2 only (4 each)    │
//  │  2'b10   │    3    │ 1 accumulated sum → Shaaban 0 slot 0 only        │
//  └──────────┴─────────┴──────────────────────────────────────────────────┘
//
// STAGE 1 ASSEMBLY — WHY 120 TAPS ARE NOT ENOUGH
//   12 trees × 10 L1 taps = 120 direct conv25 results.
//   32 Shaabans × 4 inputs = 128 inputs needed.
//   The missing 8 come from ext_sum_correction (the orphan correction layer).
//
//   flat_s1 layout (128 entries):
//   ┌──────────────────────────────────────────────────────────────────────┐
//   │ Trees 0–7  (stride 11): flat_s1[t*11 .. t*11+9]  = taps[0..9]      │
//   │                         flat_s1[t*11 + 10]        = corr_out[t]     │
//   │ Trees 8–11 (stride 10): flat_s1[88+(t-8)*10 .. ] = taps[0..9]      │
//   └──────────────────────────────────────────────────────────────────────┘
//   Shaaban s ← flat_s1[s*4 .. s*4+3]
// =============================================================================

module adder_tree_shaaban_connect #(
    // ── Adder tree ────────────────────────────────────────────────────────────
    parameter int N_TREES        = 12,
    parameter int TAPS_PER_TREE  = 10,

    // ── Shaaban unit ─────────────────────────────────────────────────────────
    parameter int N_SHAABAN      = 32,
    parameter int INPUTS_PER_SHB = 4,
    parameter int POOL_NUM       = 2,   // = INPUTS_PER_SHB / 2

    // ── Common ────────────────────────────────────────────────────────────────
    parameter int DATA_WIDTH     = 18,

    // ── Derived — do NOT override ─────────────────────────────────────────────
    parameter int TOTAL_S1_INPUTS = N_SHAABAN * INPUTS_PER_SHB,      // 128
    parameter int TOTAL_TAPS      = N_TREES * TAPS_PER_TREE,         // 120
    parameter int N_CORRECTION    = TOTAL_S1_INPUTS - TOTAL_TAPS,    // 8
    parameter int S2_ACTIVE       = 3,   // Shaaban units active in Stage 2
    parameter int S3_ACTIVE       = 1    // Shaaban units active in Stage 3
)(
    input  logic        clk,
    input  logic        rst,
    input  logic [1:0]  src_sel,   // Stage selector — see table above

    // 12 trees × 32 MAC products  [tree_index][product_index]
    input  logic signed [N_TREES-1:0][31:0][DATA_WIDTH-1:0] mac_in,

    // Stage 3 pre-accumulated 64-channel sum (reduced outside this module)
    input  logic signed [DATA_WIDTH-1:0] final_s3,

    // Shared Shaaban weights
    input  logic signed [DATA_WIDTH-1:0] conv_bias,
    input  logic signed [DATA_WIDTH-1:0] mult_weight,
    input  logic signed [DATA_WIDTH-1:0] add_weight,

    output logic [N_SHAABAN-1:0] spike_out
);

    // =========================================================================
    // SECTION 1 — 12 Adder tree instances
    // =========================================================================
    // tree_tap[t][k] = conv25_{k+1} of tree t → complete conv25 result
    // tree_final[t]  = sum of all 32 inputs of tree t → used in Stage 2/3
    // =========================================================================

    logic signed [DATA_WIDTH-1:0] tree_tap   [0:N_TREES-1][0:TAPS_PER_TREE-1];
    logic signed [DATA_WIDTH-1:0] tree_final [0:N_TREES-1];

    genvar t;
    generate
        for (t = 0; t < N_TREES; t++) begin : gen_trees
            adder_tree_10_4_1_1 u_tree (
                .in_1 (mac_in[t][0]),  .in_2 (mac_in[t][1]),  .in_3 (mac_in[t][2]),
                .in_4 (mac_in[t][3]),  .in_5 (mac_in[t][4]),  .in_6 (mac_in[t][5]),
                .in_7 (mac_in[t][6]),  .in_8 (mac_in[t][7]),  .in_9 (mac_in[t][8]),
                .in_10(mac_in[t][9]),  .in_11(mac_in[t][10]), .in_12(mac_in[t][11]),
                .in_13(mac_in[t][12]), .in_14(mac_in[t][13]), .in_15(mac_in[t][14]),
                .in_16(mac_in[t][15]), .in_17(mac_in[t][16]), .in_18(mac_in[t][17]),
                .in_19(mac_in[t][18]), .in_20(mac_in[t][19]), .in_21(mac_in[t][20]),
                .in_22(mac_in[t][21]), .in_23(mac_in[t][22]), .in_24(mac_in[t][23]),
                .in_25(mac_in[t][24]), .in_26(mac_in[t][25]), .in_27(mac_in[t][26]),
                .in_28(mac_in[t][27]), .in_29(mac_in[t][28]), .in_30(mac_in[t][29]),
                .in_31(mac_in[t][30]), .in_32(mac_in[t][31]),

                .conv25_1 (tree_tap[t][0]), .conv25_2 (tree_tap[t][1]),
                .conv25_3 (tree_tap[t][2]), .conv25_4 (tree_tap[t][3]),
                .conv25_5 (tree_tap[t][4]), .conv25_6 (tree_tap[t][5]),
                .conv25_7 (tree_tap[t][6]), .conv25_8 (tree_tap[t][7]),
                .conv25_9 (tree_tap[t][8]), .conv25_10(tree_tap[t][9]),

                .final_output(tree_final[t])
            );
        end
    endgenerate

    // =========================================================================
    // SECTION 2 — Orphan correction layer (ext_sum_correction)
    // =========================================================================
    // corr_out[c] = accurate conv25 partial sum for orphan correction c (0..7)
    // See ext_sum_correction.sv for full explanation of the grouping logic.
    // =========================================================================

    logic signed [DATA_WIDTH-1:0] corr_out [0:N_CORRECTION-1];

    ext_sum_correction #(
        .N_TREES      (N_TREES),
        .N_CORRECTION (N_CORRECTION),
        .DATA_WIDTH   (DATA_WIDTH)
    ) u_correction (
        .mac_in   (mac_in),
        .corr_out (corr_out)
    );

    // =========================================================================
    // SECTION 3 — Stage 1 flat array  (128 entries)
    // =========================================================================

    logic signed [DATA_WIDTH-1:0] flat_s1 [0:TOTAL_S1_INPUTS-1];

    genvar k;
    generate
        // Trees 0–7: 10 taps then 1 correction (stride = 11)
        for (t = 0; t < N_CORRECTION; t++) begin : gen_s1_corrected
            for (k = 0; k < TAPS_PER_TREE; k++) begin : gen_taps
                assign flat_s1[t * (TAPS_PER_TREE + 1) + k] = tree_tap[t][k];
            end
            assign flat_s1[t * (TAPS_PER_TREE + 1) + TAPS_PER_TREE] = corr_out[t];
        end

        // Trees 8–11: 10 taps only (stride = 10), no correction needed
        for (t = N_CORRECTION; t < N_TREES; t++) begin : gen_s1_standard
            for (k = 0; k < TAPS_PER_TREE; k++) begin : gen_taps_std
                assign flat_s1[N_CORRECTION * (TAPS_PER_TREE + 1)
                               + (t - N_CORRECTION) * TAPS_PER_TREE + k]
                       = tree_tap[t][k];
            end
        end
    endgenerate

    // =========================================================================
    // SECTION 4 — 32 Shaaban units with 3-way input MUX
    // =========================================================================

    genvar s, p;
    generate
        for (s = 0; s < N_SHAABAN; s++) begin : gen_shaaban

            logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] src_s1;
            logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] src_s2;
            logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] src_s3;
            logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_conv_in;

            for (p = 0; p < INPUTS_PER_SHB; p++) begin : gen_sources

                // Stage 1: 4 consecutive slots from flat_s1
                assign src_s1[p*DATA_WIDTH +: DATA_WIDTH] =
                       flat_s1[s * INPUTS_PER_SHB + p];

                // Stage 2: 4 tree finals per active Shaaban; zero for inactive
                if (s < S2_ACTIVE)
                    assign src_s2[p*DATA_WIDTH +: DATA_WIDTH] =
                           tree_final[s * INPUTS_PER_SHB + p];
                else
                    assign src_s2[p*DATA_WIDTH +: DATA_WIDTH] = '0;

                // Stage 3: only Shaaban 0 slot 0 is live; all others zero
                if (s == 0 && p == 0)
                    assign src_s3[p*DATA_WIDTH +: DATA_WIDTH] = final_s3;
                else
                    assign src_s3[p*DATA_WIDTH +: DATA_WIDTH] = '0;

            end

            // Combinational 3-to-1 MUX
            always_comb begin
                unique case (src_sel)
                    2'b00:   shb_conv_in = src_s1;
                    2'b01:   shb_conv_in = src_s2;
                    2'b10:   shb_conv_in = src_s3;
                    default: shb_conv_in = '0;
                endcase
            end

            shaban_unit_top #(
                .DATA_WIDTH        (DATA_WIDTH),
                .conv_bias_relu_num(INPUTS_PER_SHB),
                .batch_norm_num    (INPUTS_PER_SHB),
                .pool_num          (POOL_NUM)
            ) u_shaaban (
                .clk        (clk),
                .rst        (rst),
                .conv_in    (shb_conv_in),
                .conv_bias  (conv_bias),
                .mult_wight (mult_weight),
                .add_wight  (add_weight),
                .spike      (spike_out[s])
            );

        end
    endgenerate

endmodule
