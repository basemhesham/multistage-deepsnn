`timescale 1ns / 1ps
// =============================================================================
// conv9_adder.sv
// -----------------------------------------------------------------------------
// 4-level binary adder tree: reduces 9 × 36-bit products to one 40-bit sum.
//
// TREE STRUCTURE
//   Level 1 (pairs):      m[0]+m[1], m[2]+m[3], m[4]+m[5], m[6]+m[7]  → 37-bit
//   Level 2 (pairs):      s1_0+s1_1, s1_2+s1_3                         → 38-bit
//   Level 3 (pair):       s2_0+s2_1                                     → 39-bit
//   Level 4 (final):      s3 + m[8]                                     → 40-bit
//
//   m[8] bypasses levels 1-3 and is added last to keep the tree balanced.
//   Bit widths grow by 1 per level to prevent overflow.
//
// PORTS
//   m [0:8][35:0]  — 9 products from conv9_mul (36-bit each)
//   Pixel_Out [39:0] — final 40-bit dot product sum
//
// PURELY COMBINATIONAL — no clock, no state.
// USED BY: cov9.sv
// =============================================================================

module conv9_adder #(
    parameter int PROD_W = 36,   // input product width
    parameter int OUT_W  = 40    // output sum width
)(
    input  wire [PROD_W-1:0] m [0:8],     // 9 products
    output logic [OUT_W-1:0] Pixel_Out    // accumulated sum
);

    // ── Level 1: 4 pair sums (36-bit + 36-bit → 37-bit) ─────────────────────
    wire [36:0] s1_0 = m[0] + m[1];
    wire [36:0] s1_1 = m[2] + m[3];
    wire [36:0] s1_2 = m[4] + m[5];
    wire [36:0] s1_3 = m[6] + m[7];
    // m[8] passes through directly to Level 4

    // ── Level 2: 2 pair sums (37-bit + 37-bit → 38-bit) ─────────────────────
    wire [37:0] s2_0 = s1_0 + s1_1;
    wire [37:0] s2_1 = s1_2 + s1_3;

    // ── Level 3: 1 pair sum  (38-bit + 38-bit → 39-bit) ─────────────────────
    wire [38:0] s3 = s2_0 + s2_1;

    // ── Level 4: add m[8] bypass (39-bit + 36-bit → 40-bit) ─────────────────
    assign Pixel_Out = s3 + m[8];

endmodule