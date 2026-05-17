`timescale 1ns / 1ps
// =============================================================================
// adder_layer1.v  -  3-input adder using DSP48E1 primitive (Virtex-7)
// Operation: P = (A+D) + C   (pre-adder: A+D, then +C via Z mux)
// Input:  3 x 18-bit signed
// Output: 20-bit signed
// =============================================================================

module adder_layer1 #(
    parameter IN_WIDTH  = 18,
    parameter OUT_WIDTH = 20
)(
    input  wire signed [IN_WIDTH-1:0]  add_1,   // ? A port
    input  wire signed [IN_WIDTH-1:0]  add_2,   // ? C port
    input  wire signed [IN_WIDTH-1:0]  add_3,   // ? D port
    output wire signed [OUT_WIDTH-1:0] adder_out
);

    wire signed [47:0] P_full;
/*
--------------------------------------------------------------------------------------
When use STM Board Change Follwoing:
    // 1. Primitive name:
        DSP48E1  →  DSP48E2
    // 2. D port width:
    .D (25'b0)  →  .D (27'b0
    // 3. Remove this line entirely:
    .USE_DPORT ("FALSE")   // ← delete
    .USE_DPORT ("TRUE")    // ← delete (adder_layer1/2/3 use D port)
----------------------------------------------------------------------------------------------
*/


    DSP48E1 #(
        .AREG       (0),
        .BREG       (0),
        .CREG       (0),
        .DREG       (0),
        .MREG       (0),
        .PREG       (0),
        .ADREG      (0),
        .A_INPUT    ("DIRECT"),
        .B_INPUT    ("DIRECT"),
        .USE_MULT   ("NONE"),
        .USE_SIMD   ("ONE48"),
        .USE_DPORT  ("TRUE")
    ) dsp_inst (
        .A      ({{(30-IN_WIDTH){add_1[IN_WIDTH-1]}}, add_1}),
        .D      ({{(25-IN_WIDTH){add_3[IN_WIDTH-1]}}, add_3}),
        .C      ({{(48-IN_WIDTH){add_2[IN_WIDTH-1]}}, add_2}),
        .B      (18'b0),
        .PCIN   (48'b0),

        // Z=011(C), XY=0011(A+D via pre-adder) ? P = C + (A+D)
        .OPMODE     (7'b0110011),
        .ALUMODE    (4'b0000),
        .INMODE     (5'b00000),
        .CARRYINSEL (3'b000),
        .CARRYIN    (1'b0),

        .P      (P_full),
        .PCOUT  (),

        .CLK        (1'b0),
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