`timescale 1ns / 1ps

module adder_layer1 #(
    parameter IN_WIDTH  = 18,
    parameter OUT_WIDTH = 20
    )(
    input  wire signed [IN_WIDTH-1:0] add_1,
    input  wire signed [IN_WIDTH-1:0] add_2,
    input  wire signed [IN_WIDTH-1:0] add_3,

    output wire signed [OUT_WIDTH-1:0] adder_out
    );


    // (A+D) + C
    dsp48_layer_1   dsp_unit_layer_1 (
        (add_1),      // input wire [17 : 0] A
        (add_2),      // input wire [17 : 0] C
        (add_3),      // input wire [17 : 0] D
        (adder_out)   // output wire [19 : 0] P
    );

endmodule
