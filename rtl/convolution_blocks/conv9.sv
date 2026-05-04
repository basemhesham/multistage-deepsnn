`timescale 1ns / 1ps
// =============================================================================
// conv9_array.sv
// -----------------------------------------------------------------------------
// Instantiates 384 cov9 units organised as:
//
//   N_GROUPS  = 12   (one group per adder_tree_10_4_1_1)
//   N_CH      = 32   (one unit per input channel inside a group)
//   Total     = 12 × 32 = 384 cov9 instances
//
// WHY THIS SHAPE?
// ───────────────
// Stage 2 computes Conv2: 3×3, 32 input channels → 64 output channels.
// For one output pixel of one output channel the hardware must form:
//
//   out[oc] = Σ_{ic=0}^{31}  conv9( pixels[ic][3×3],  weights[oc][ic][3×3] )
//
// Each cov9 computes the 9-element dot product for ONE input channel (ic).
// The 32 results go into ONE adder tree which sums them to give ONE output
// channel value for ONE spatial position.
//
// 12 groups run in parallel → 12 output values per cycle.
// The 64 output channels (and all spatial positions) are handled by
// time-multiplexing the weight ROM address — this module always computes
// 12 × 32 = 384 dot products simultaneously, wired combinationally.
//
// DATA-FLOW DIAGRAM (one group g, output channel driven by current ROM addr)
// ─────────────────────────────────────────────────────────────────────────
//
//   pixels[g][0][0:8] ──► cov9[g][0] ──► mac_out[g][0]  ─┐
//   pixels[g][1][0:8] ──► cov9[g][1] ──► mac_out[g][1]   │
//   pixels[g][2][0:8] ──► cov9[g][2] ──► mac_out[g][2]   │ 32 products
//   ...                                                    │ into adder tree g
//   pixels[g][31][0:8] ─► cov9[g][31] ─► mac_out[g][31] ─┘
//
//   (adder_tree_10_4_1_1 takes these 32 values → 1 accumulated 18-bit sum)
//
// PORTS
// ─────
//   pixels   [N_GROUPS][N_CH][9][PIXEL_W-1:0]
//            Pixel values. pixels[g][ic][k] = k-th element of the 3×3 window
//            for input channel ic, belonging to group g.
//            Supplied by the conv_input_mux / frame_input_mapping module.
//
//   weights  [N_GROUPS][N_CH][9][PIXEL_W-1:0]
//            Weight values. weights[g][ic][k] = k-th weight of the 3×3 kernel
//            for input channel ic of the *current* output channel.
//            All 12 groups share the same output-channel ROM address at once;
//            only the spatial-position grouping (g) and input channel (ic) differ.
//
//   mac_out  [N_GROUPS][N_CH][OUT_W-1:0]
//            40-bit dot product results. Connect directly to mac_in[g][ic] of
//            the 12 adder_tree_10_4_1_1 instances in adder_tree_shaaban_connect.
//
// PURELY COMBINATIONAL — no clk/rst needed in this module.
// =============================================================================

module conv9_array #(
    parameter int N_GROUPS = 12,    // = number of adder trees
    parameter int N_CH     = 32,    // = number of input channels (Stage 2)
    parameter int PIXEL_W  = 18,    // data width
    parameter int PROD_W   = 36,    // multiplier product width (internal to cov9)
    parameter int OUT_W    = 40     // adder-tree output width
)(
    // ── Pixel inputs: [group][input_channel][tap_0..8] ──────────────────────
    input  wire logic [PIXEL_W-1:0] pixels  [0:N_GROUPS-1][0:N_CH-1][0:8],

    // ── Weight inputs: [group][input_channel][tap_0..8] ─────────────────────
    // (all groups share the same output-channel address; only ic index differs)
    input  wire logic [PIXEL_W-1:0] weights [0:N_GROUPS-1][0:N_CH-1][0:8],

    // ── 40-bit dot-product outputs: [group][input_channel] ──────────────────
    // Wire directly to mac_in[g][ic] of the adder tree array.
    output logic      [OUT_W-1:0]   mac_out [0:N_GROUPS-1][0:N_CH-1]
);

    // =========================================================================
    // Generate: 12 groups × 32 units = 384 cov9 instances
    //
    //   Outer loop  g  = group index  (0 .. N_GROUPS-1  = 0..11)
    //   Inner loop  ic = input-channel index (0 .. N_CH-1  = 0..31)
    //
    //   Instance name:  gen_groups[g].gen_ch[ic].u_conv9
    //
    //   Connections:
    //     P  ← pixels [g][ic][0:8]   (nine 18-bit pixel taps)
    //     Q  ← weights[g][ic][0:8]   (nine 18-bit weight values)
    //     Pixel_Out → mac_out[g][ic]  (40-bit accumulated product)
    // =========================================================================

    genvar g, ic;
    generate
        for (g = 0; g < N_GROUPS; g++) begin : gen_groups
            for (ic = 0; ic < N_CH; ic++) begin : gen_ch

                cov9 #(
                    .PIXEL_W (PIXEL_W),
                    .PROD_W  (PROD_W),
                    .OUT_W   (OUT_W)
                ) u_conv9 (
                    .P         (pixels [g][ic]),   // wire [PIXEL_W-1:0] P [0:8]
                    .Q         (weights[g][ic]),   // wire [PIXEL_W-1:0] Q [0:8]
                    .Pixel_Out (mac_out [g][ic])   // logic [OUT_W-1:0]
                );

            end
        end
    endgenerate

endmodule