`timescale 1ns / 1ps
// =============================================================================
// conv9.sv
// -----------------------------------------------------------------------------
// Single conv9 unit: computes a signed 9-element dot product.
//
//   Pixel_Out = sum_{k=0}^{8}  P[k] * Q[k]
//
// PORTS
//   P [0:8][17:0]    - 9 signed pixel  values
//   Q [0:8][17:0]    - 9 signed weight values
//   Pixel_Out [39:0] - 40-bit signed dot product result
//
// PURELY COMBINATIONAL.
// =============================================================================

module conv9 #(
    parameter int PIXEL_W = 18,
    parameter int PROD_W  = 36,
    parameter int OUT_W   = 40
)(
    input  wire logic signed [PIXEL_W-1:0] P [0:8],
    input  wire logic signed [PIXEL_W-1:0] Q [0:8],
    output logic      signed [OUT_W-1:0]   Pixel_Out
);

    // Internal wire: 9 products from the multiplier stage
    wire signed [PROD_W-1:0] m [0:8];

    // Signed intermediate arrays to avoid type mismatch on conv9_mul ports
    // (conv9_mul expects unsigned bit vectors; we cast explicitly here)
    wire logic signed [PIXEL_W-1:0] P_bits [0:8];
    wire logic signed [PIXEL_W-1:0] Q_bits [0:8];

    genvar k;
    generate
        for (k = 0; k < 9; k++) begin : gen_cast
            assign P_bits[k] = P[k];
            assign Q_bits[k] = Q[k];
        end
    endgenerate

    // Stage 1: 9 signed multipliers
    conv9_mul #(
        .PIXEL_W (PIXEL_W),
        .PROD_W  (PROD_W)
    ) u_mul (
        .P (P_bits),
        .Q (Q_bits),
        .m (m)
    );

    // Stage 2: signed adder tree -> final dot product
    conv9_adder #(
        .PROD_W (PROD_W),
        .OUT_W  (OUT_W)
    ) u_adder (
        .m         (m),
        .Pixel_Out (Pixel_Out)
    );

endmodule