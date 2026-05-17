`timescale 1ns / 1ps
// =============================================================================
// adder_layer4.v  -  2-input adder using DSP48E1 (Virtex-7)
// Operation: P = A + C   (no D port needed for 2 inputs)
// Input:  add_1=24-bit, add_2=22-bit signed
// Output: 25-bit signed
// =============================================================================

module adder_layer4 #(
    parameter IN1_WIDTH = 24,
    parameter IN2_WIDTH = 22,
    parameter OUT_WIDTH = 25
)(
    input  wire signed [IN1_WIDTH-1:0] add_1,   // → A port (wider)
    input  wire signed [IN2_WIDTH-1:0] add_2,   // → C port
    output wire signed [OUT_WIDTH-1:0] adder_out
);

    wire signed [47:0] P_full;

    DSP48E1 #(
        .AREG       (0), .BREG(0), .CREG(0), .DREG(0),
        .MREG       (0), .PREG(0), .ADREG(0),
        .A_INPUT    ("DIRECT"), .B_INPUT("DIRECT"),
        .USE_MULT   ("NONE"), .USE_SIMD("ONE48"),
        .USE_DPORT  ("FALSE")   // only 2 inputs - no D port needed
    ) dsp_inst (
        .A      ({{(30-IN1_WIDTH){add_1[IN1_WIDTH-1]}}, add_1}),
        .C      ({{(48-IN2_WIDTH){add_2[IN2_WIDTH-1]}}, add_2}),
        .D      (25'b0), .B(18'b0), .PCIN(48'b0),

        // Z=011(C), XY=0000(0) but we want Z+A
        // OPMODE: Z=011(C), X=00(0), Y=00(0) + use ALUMODE with A
        // Simpler: Z=011(C), XY=0010(A:B→just A path via OPMODE X=10)
        // P = C + A:  OPMODE = 7'b0110010 (Z=C, X=A, Y=0)
        .OPMODE(7'b0110010),
        .ALUMODE(4'b0000),
        .INMODE(5'b00000), .CARRYINSEL(3'b000), .CARRYIN(1'b0),

        .P(P_full), .PCOUT(),
        .CLK(1'b0),
        .CEA1(1'b1),.CEA2(1'b1),.CEB1(1'b1),.CEB2(1'b1),
        .CEC(1'b1),.CED(1'b1),.CEM(1'b1),.CEP(1'b1),
        .CEAD(1'b1),.CECTRL(1'b1),.CECARRYIN(1'b1),
        .RSTA(1'b0),.RSTB(1'b0),.RSTC(1'b0),.RSTD(1'b0),
        .RSTM(1'b0),.RSTP(1'b0),.RSTALLCARRYIN(1'b0),
        .RSTALUMODE(1'b0),.RSTCTRL(1'b0),.RSTINMODE(1'b0),
        .ACIN(30'b0),.BCIN(18'b0),
        .CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
        .CARRYOUT(),.CARRYCASCOUT(),.MULTSIGNOUT(),
        .OVERFLOW(),.UNDERFLOW(),
        .PATTERNDETECT(),.PATTERNBDETECT()
    );

    assign adder_out = P_full[OUT_WIDTH-1:0];

endmodule