`timescale 1ns / 1ps

module adder_3(
    input  wire signed [17:0] add_1,
    input  wire signed [17:0] add_2,
    input  wire signed [17:0] add_3,

    output wire signed [17:0] final_adder_out,
    output wire signed [47:0] internal_adder_out
    );

    // Sign-extend all operands to the internal precision before summing.
    wire signed [47:0] sum_full;
    assign sum_full = $signed(add_1) + $signed(add_2) + $signed(add_3);

    // Internal high-precision path (similar intent to DSP PCOUT).
    assign internal_adder_out = sum_full;

    // 18-bit output path (truncates to keep the original module interface).
    assign final_adder_out = sum_full[17:0];

endmodule
