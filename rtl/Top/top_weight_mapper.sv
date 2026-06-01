`timescale 1ns / 1ps

module top_weight_mapper #(
    parameter int PIXEL_W = 18
)(
    input  logic [1:0]                src_sel,
    input  logic [5:0]                conv2_filter,
    input  logic [6:0]                conv3_filter,
    output logic signed [PIXEL_W-1:0] weights_mapped [0:11][0:31][0:8]
);

    logic [PIXEL_W-1:0] stage1_weights [3456];
    logic [PIXEL_W-1:0] stage2_weights [3456];
    logic [PIXEL_W-1:0] stage3_weights [3456];
    logic [PIXEL_W-1:0] active_weights [3456];

    CONV1_W_MAP_OPT u_w1 (
        .conv9_in (stage1_weights)
    );

    CONV2_W_MAP_OPT u_w2 (
        .filter   (conv2_filter),
        .conv9_in (stage2_weights)
    );

    CONV3_W_MAP_OPT u_w3 (
        .filter   (conv3_filter),
        .conv9_in (stage3_weights)
    );

    always_comb begin
        case (src_sel)
            2'b00:   active_weights = stage1_weights;
            2'b01:   active_weights = stage2_weights;
            2'b10:   active_weights = stage3_weights;
            default: active_weights = stage2_weights;
        endcase
    end

    genvar gw, cw, tw;
    generate
        for (gw = 0; gw < 12; gw++) begin : gen_wmap_row
            for (cw = 0; cw < 32; cw++) begin : gen_wmap_col
                for (tw = 0; tw < 9; tw++) begin : gen_wmap_tap
                    assign weights_mapped[gw][cw][tw] =
                        $signed(active_weights[(gw * 32 + cw) * 9 + tw]);
                end
            end
        end
    endgenerate

endmodule
