`timescale 1ns / 1ps
// =============================================================================
// adder_layer4.v - 2-input adder inferred into DSP48E2 by Vivado
// Operation: P = add_1 + add_2
// Input:  add_1=24-bit, add_2=22-bit signed
// Output: 25-bit signed
// =============================================================================

(* use_dsp = "yes" *)
module adder_layer4 #(
    parameter IN1_WIDTH = 24,
    parameter IN2_WIDTH = 22,
    parameter OUT_WIDTH = 25
)(
    input  wire signed [IN1_WIDTH-1:0] add_1,
    input  wire signed [IN2_WIDTH-1:0] add_2,
    output wire signed [OUT_WIDTH-1:0] adder_out
);

    wire signed [47:0] add_1_ext;
    wire signed [47:0] add_2_ext;
    (* use_dsp = "yes" *) wire signed [47:0] P_full;

    assign add_1_ext = {{(48-IN1_WIDTH){add_1[IN1_WIDTH-1]}}, add_1};
    assign add_2_ext = {{(48-IN2_WIDTH){add_2[IN2_WIDTH-1]}}, add_2};
    assign P_full    = add_1_ext + add_2_ext;

    assign adder_out = P_full[OUT_WIDTH-1:0];

endmodule
