`timescale 1ns / 1ps
// =============================================================================
// xbip_dsp48_macro_cascade.sv
// Virtex-7 compatible cascade wrapper using DSP48E1
// Operation: PCOUT = A*B + PCIN
// First DSP in chain: pass PCIN=48'b0 from conv9.sv
// =============================================================================

(* use_dsp = "yes" *)
module xbip_dsp48_macro_cascade #(
    parameter int PIXEL_W = 18
)(
    input  logic                       CLK,
    input  logic signed [PIXEL_W-1:0]  A,      // pixel
    input  logic signed [PIXEL_W-1:0]  B,      // weight
    input  logic signed [47:0]         PCIN,   // cascade in (48-bit)
    output logic signed [47:0]         PCOUT,  // cascade out → next DSP's PCIN only
    output logic signed [47:0]         P_fab   // fabric output → use for last DSP in chain
);

    // DSP48E1 A port = 30-bit, B port = 18-bit
    logic signed [29:0] A_ext;
    assign A_ext = {{(30-PIXEL_W){A[PIXEL_W-1]}}, A};  // sign-extend

    (* keep = "true" *)
    DSP48E1 #(
        .AREG           (0),        // no input register on A
        .BREG           (0),        // no input register on B
        .CREG           (0),
        .DREG           (0),
        .MREG           (0),        // no multiplier register
        .PREG           (1),        // register P output - keeps CLK alive
        .ADREG          (0),
        .A_INPUT        ("DIRECT"),
        .B_INPUT        ("DIRECT"),
        .USE_MULT       ("MULTIPLY"),
        .USE_SIMD       ("ONE48"),
        .USE_DPORT      ("FALSE")   // valid in DSP48E1
    ) dsp_inst (
        // Data ports
        .A              (A_ext),    // 30-bit
        .B              (B),        // 18-bit
        .C              (48'b0),
        .D              (25'b0),    // 25-bit in DSP48E1 (not 27)
        .PCIN           (PCIN),     // cascade input

        // Operation
        // OPMODE 7'b0010101:
        //   Z mux  = 001 → PCIN
        //   XY mux = 0101 → M (A*B)
        //   Result: P = A*B + PCIN
        .OPMODE         (7'b0010101),
        .ALUMODE        (4'b0000),
        .INMODE         (5'b00000),
        .CARRYINSEL     (3'b000),
        .CARRYIN        (1'b0),

        // Cascade output - drives PCIN of next DSP only (not fabric)
        .PCOUT          (PCOUT),
        // Fabric output - use this for the last DSP in the chain
        .P              (P_fab),

        // Clock and enables
        .CLK            (CLK),
        .CEA1           (1'b1),
        .CEA2           (1'b1),
        .CEB1           (1'b1),
        .CEB2           (1'b1),
        .CEC            (1'b1),
        .CED            (1'b1),
        .CEM            (1'b1),
        .CEP            (1'b1),
        .CEAD           (1'b1),
        .CECTRL         (1'b1),
        .CECARRYIN      (1'b1),

        // Resets - all tied low (no reset)
        .RSTA           (1'b0),
        .RSTB           (1'b0),
        .RSTC           (1'b0),
        .RSTD           (1'b0),
        .RSTM           (1'b0),
        .RSTP           (1'b0),
        .RSTALLCARRYIN  (1'b0),
        .RSTALUMODE     (1'b0),
        .RSTCTRL        (1'b0),
        .RSTINMODE      (1'b0),

        // Unused cascade ports
        .ACIN           (30'b0),
        .BCIN           (18'b0),
        .CARRYCASCIN    (1'b0),
        .MULTSIGNIN     (1'b0),

        // Unused outputs
        .CARRYOUT       (),
        .CARRYCASCOUT   (),
        .MULTSIGNOUT    (),
        .OVERFLOW       (),
        .UNDERFLOW      (),
        .PATTERNDETECT  (),
        .PATTERNBDETECT ()
    );

endmodule