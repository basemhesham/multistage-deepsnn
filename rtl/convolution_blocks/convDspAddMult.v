`timescale 1ns / 1ps
// =============================================================================
// xbip_dsp48_macro_cascade.sv
// Vivado/XCVU11P DSP wrapper.
// Operation: PCOUT = A*B + PCIN
//
// Keep the wrapper interface stable and infer the target DSP primitive; Vivado
// maps this arithmetic to DSP48E2 slices for UltraScale+ devices.
// =============================================================================

(* use_dsp = "yes" *)
module xbip_dsp48_macro_cascade #(
    parameter int PIXEL_W = 18
)(
    input  logic                       CLK,
    input  logic signed [PIXEL_W-1:0]  A,
    input  logic signed [PIXEL_W-1:0]  B,
    input  logic signed [47:0]         PCIN,
    output logic signed [47:0]         PCOUT,
    output logic signed [47:0]         P_fab
);

    localparam int PRODUCT_W = 2 * PIXEL_W;

    (* use_dsp = "yes" *) logic signed [PRODUCT_W-1:0] product;
    logic signed [47:0] product_ext;
    logic signed [47:0] result;

    always_comb begin
        product     = A * B;
        product_ext = {{(48-PRODUCT_W){product[PRODUCT_W-1]}}, product};
        result      = product_ext + PCIN;
    end

    assign PCOUT = result;
    assign P_fab = result;

endmodule
