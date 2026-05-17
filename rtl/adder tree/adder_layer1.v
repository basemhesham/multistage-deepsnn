`timescale 1ns / 1ps
// =============================================================================
// adder_layer1.v - 3-input adder inferred into DSP48E2 by Vivado
// Operation: P = add_1 + add_2 + add_3
// Input:  3 x 18-bit signed
// Output: 20-bit signed
// =============================================================================

(* use_dsp = "yes" *)
module adder_layer1 #(
    parameter IN_WIDTH  = 18,
    parameter OUT_WIDTH = 20
)(
    input  wire signed [IN_WIDTH-1:0]  add_1,
    input  wire signed [IN_WIDTH-1:0]  add_2,
    input  wire signed [IN_WIDTH-1:0]  add_3,
    output wire signed [OUT_WIDTH-1:0] adder_out
);

    wire signed [47:0] add_1_ext;
    wire signed [47:0] add_2_ext;
    wire signed [47:0] add_3_ext;
    (* use_dsp = "yes" *) wire signed [47:0] P_full;

    assign add_1_ext = {{(48-IN_WIDTH){add_1[IN_WIDTH-1]}}, add_1};
    assign add_2_ext = {{(48-IN_WIDTH){add_2[IN_WIDTH-1]}}, add_2};
    assign add_3_ext = {{(48-IN_WIDTH){add_3[IN_WIDTH-1]}}, add_3};
    assign P_full    = add_1_ext + add_2_ext + add_3_ext;

    assign adder_out = P_full[OUT_WIDTH-1:0];

endmodule
