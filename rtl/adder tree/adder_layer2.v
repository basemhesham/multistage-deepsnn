`timescale 1ns / 1ps

module adder_layer2 #(
    parameter IN_WIDTH  = 20,
    parameter OUT_WIDTH = 22
    )(
    input  wire signed [IN_WIDTH-1:0] add_1,
    input  wire signed [IN_WIDTH-1:0] add_2,
    input  wire signed [IN_WIDTH-1:0] add_3,

    output wire signed [OUT_WIDTH-1:0] adder_out
    );


    // (A+D)*B + C
    dsp48_layer_2   dsp_unit_layer2 (
        .A(add_1),           // input wire [20 : 0] A
        .C(add_2),           // input wire [20 : 0] C
        .D(add_3),           // input wire [20 : 0] D
        .P(adder_out)        // output wire [22 : 0] P
    );

endmodule
