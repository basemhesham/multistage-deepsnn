`timescale 1ns / 1ps
// =============================================================================
// Batch_Norm.v  -  Batch Normalisation using DSP48E1 (Virtex-7)
// Operation: P = (Batch_Norm_in � mult_wight) + add_wight
// Input:  3 x 18-bit signed
// Output: 18-bit signed  (bits [36:19] of the 48-bit MAC result)
//         i.e. the result is scaled back to 18-bit fixed-point
// =============================================================================

module Batch_Norm #(
    parameter DATA_WIDTH = 18
)(
    input  wire signed [DATA_WIDTH-1:0] Batch_Norm_in,  // ? A port
    input  wire signed [DATA_WIDTH-1:0] mult_wight,     // ? B port
    input  wire signed [DATA_WIDTH-1:0] add_wight,      // ? C port
    output wire signed [DATA_WIDTH-1:0] Batch_Norm_out
);

    wire signed [47:0] P_full;

    /* OLD VERSIOM USING Xilinx IP (commented out for now, as we are using DSP48E1 primitive directly)
    xbip_dsp48_macro_0 dsp48_inst_0 (
    .A(Batch_Norm_in),  // input wire [17 : 0] A
    .B(mult_wight),  // input wire [17 : 0] B
    .C(add_wight),  // input wire [17 : 0] C
    .P(mult_Add_result)  // output wire [36 : 0] P
    */


    // Output scaling: take bits [36:19] of 48-bit MAC result
    // This matches original xbip behaviour: P[36:19] ? 18-bit fixed-point
    assign Batch_Norm_out = P_full[36:19];

/*-------------------------------------------------------------------
    When use STM Board Change Follwoing: 
   1- DSP48E1 → DSP48E2
   2-.D (25'b0)  → .D (27'b0)
----------------------------------------------------------------------------------*/

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
        .USE_MULT   ("MULTIPLY"),   // A*B
        .USE_SIMD   ("ONE48"),
        .USE_DPORT  ("FALSE")
    ) dsp_inst (
        // A = Batch_Norm_in, sign-extended to 30-bit
        .A      ({{(30-DATA_WIDTH){Batch_Norm_in[DATA_WIDTH-1]}}, Batch_Norm_in}),

        // B = mult_wight (already 18-bit)
        .B      (mult_wight),

        // C = add_wight, sign-extended to 48-bit
        .C      ({{(48-DATA_WIDTH){add_wight[DATA_WIDTH-1]}}, add_wight}),

        .D      (25'b0),
        .PCIN   (48'b0),

        // OPMODE: Z=011(C), XY=0101(M=A*B) ? P = A*B + C
        .OPMODE     (7'b0110101),
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

endmodule