`timescale 1ns / 1ps
// =============================================================================
// conv9_mul.sv
// -----------------------------------------------------------------------------
// 9 parallel multipliers: one per pixel-weight pair.
//
// Each multiplier computes an 18-bit × 18-bit unsigned product → 36-bit.
// All 9 multiplications are purely combinational.
//
// PORTS
//   P [0:8][17:0]  — 9 pixel  values  (18-bit each)
//   Q [0:8][17:0]  — 9 weight values  (18-bit each)
//   m [0:8][35:0]  — 9 products        (36-bit each)
//
// USED BY: cov9.sv
// =============================================================================

module conv9_mul #(
    parameter int PIXEL_W = 18,         // input bit width
    parameter int PROD_W  = PIXEL_W * 2 // product bit width = 36
)(
    input  wire logic [PIXEL_W-1:0] P [0:8],   // pixel  inputs
    input  wire logic [PIXEL_W-1:0] Q [0:8],   // weight inputs
    output logic      [PROD_W-1:0]  m [0:8]    // products
);

    genvar i;
    generate
        for (i = 0; i < 9; i++) begin : MULT_STAGE
            assign m[i] = P[i] * Q[i];
        end
    endgenerate

endmodule