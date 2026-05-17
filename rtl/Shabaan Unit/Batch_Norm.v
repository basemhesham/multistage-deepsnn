`timescale 1ns / 1ps
// =============================================================================
// Batch_Norm.v - Batch normalisation inferred into DSP48E2 by Vivado
// Operation: P = (Batch_Norm_in * mult_wight) + add_wight
// Output: bits [36:19] of the 48-bit MAC result
// =============================================================================

(* use_dsp = "yes" *)
module Batch_Norm #(
    parameter DATA_WIDTH = 18
)(
    input  wire signed [DATA_WIDTH-1:0] Batch_Norm_in,
    input  wire signed [DATA_WIDTH-1:0] mult_wight,
    input  wire signed [DATA_WIDTH-1:0] add_wight,
    output wire signed [DATA_WIDTH-1:0] Batch_Norm_out
);

    localparam int PRODUCT_W = 2 * DATA_WIDTH;

    (* use_dsp = "yes" *) wire signed [PRODUCT_W-1:0] mult_result;
    wire signed [47:0] mult_ext;
    wire signed [47:0] add_ext;
    wire signed [47:0] P_full;

    assign mult_result = Batch_Norm_in * mult_wight;
    assign mult_ext    = {{(48-PRODUCT_W){mult_result[PRODUCT_W-1]}}, mult_result};
    assign add_ext     = {{(48-DATA_WIDTH){add_wight[DATA_WIDTH-1]}}, add_wight};
    assign P_full      = mult_ext + add_ext;

    assign Batch_Norm_out = P_full[36:19];

endmodule
