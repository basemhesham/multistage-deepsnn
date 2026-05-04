module main_circuit_top #(
    parameter DATA_WIDTH = 18
)(
    input  wire                         clk,
    input  wire                         rst,

    // MUX select: 2'b00=image, 2'b01=lef1, 2'b10=lef2
    input  wire [1:0]                   src_sel,
    input  wire                         arst_n,
    input  wire                         enable,
    input  wire [3199:0]                mem,

    // Shared weights
    input  wire signed [DATA_WIDTH-1:0] conv_bias,
    input  wire signed [DATA_WIDTH-1:0] mult_wight,
    input  wire signed [DATA_WIDTH-1:0] add_wight,

    // 3 spike outputs
    output wire                         spike [0:2]
);

    // =========================================================
    // INTERNAL SOURCE WIRES
    // (driven by upstream instantiations, not input ports)
    // =========================================================

    // Source 1: image frame
    wire [DATA_WIDTH-1:0] P_img  [0:383][0:8];
    wire [DATA_WIDTH-1:0] Q_img  [0:383][0:8];

    // Source 2: lef1
    wire [DATA_WIDTH-1:0] P_lef1 [0:383][0:8];
    wire [DATA_WIDTH-1:0] Q_lef1 [0:383][0:8];

    

    // Source 3: lef2
    wire [DATA_WIDTH-1:0] P_lef2 [0:383][0:8];
    wire [DATA_WIDTH-1:0] Q_lef2 [0:383][0:8];

    // TODO: replace with actual upstream module instantiations
    // image_source u_img  (.P_out(P_img),  .Q_out(Q_img),  ...);
    // lef1_source  u_lef1 (.P_out(P_lef1), .Q_out(Q_lef1), ...);
    // lef2_source  u_lef2 (.P_out(P_lef2), .Q_out(Q_lef2), ...);

    // =========================================================
    // MUX: Select input source — 395 instances x 9 taps
    // =========================================================
    wire [DATA_WIDTH-1:0] P_mux [0:383][0:8];
    wire [DATA_WIDTH-1:0] Q_mux [0:383][0:8];

    genvar idx, tap;
    generate
        for (idx = 0; idx < 384; idx = idx + 1) begin : MUX_INST
            for (tap = 0; tap < 9; tap = tap + 1) begin : MUX_TAP
                assign P_mux[idx][tap] = (src_sel == 2'b00) ? P_img [idx][tap] :
                                         (src_sel == 2'b01) ? P_lef1[idx][tap] :
                                                              P_lef2[idx][tap];
                assign Q_mux[idx][tap] = (src_sel == 2'b00) ? Q_img [idx][tap] :
                                         (src_sel == 2'b01) ? Q_lef1[idx][tap] :
                                                              Q_lef2[idx][tap];
            end
        end
    endgenerate

    // =========================================================
    // STAGE 1a: 395x cov9
    // =========================================================
    wire [39:0] Pixel_Out [0:383];

    genvar i;
    generate
        for (i = 0; i < 384; i = i + 1) begin : COV9_INST
            cov9 u_cov9 (
                .P         (P_mux[i]),
                .Q         (Q_mux[i]),
                .Pixel_Out (Pixel_Out[i])
            );
        end
    endgenerate

    // =========================================================
    // STAGE 1→2 BRIDGE: Truncate 40-bit cov9 output to 18-bit
    //   Using bits [17:0] — adjust slice if needed (e.g. [35:18])
    // =========================================================
    wire signed [DATA_WIDTH-1:0] cov9_to_adder [0:383];

    genvar j;
    generate
        for (j = 0; j < 384; j = j + 1) begin : TRUNC
            assign cov9_to_adder[j] = Pixel_Out[j][17:0];
        end
    endgenerate

    wire fil_in [31:0][39:0];
    wire conv_in [0:31][0:11][0:8];

    mem_mapping u_mem_mapping (
        .clk(clk),
        .arst_n(arst_n),
        .frame(frame_counter),
        .mem(mem),
        .fil_in(fil_in)
    );

    genvar map_k;
    generate
        for (map_k = 0; map_k <= 31; map_k = map_k + 1) begin : FRAME_INPUT_MAPPING_INST
            frame_input_mapping u_frame_input_mapping (
                .frame(frame_counter),
                .in(fil_in[map_k]),
                .conv(conv_in[map_k])
            );
        end
    endgenerate

    genvar lef_f, lef_d, lef_t;
    generate
        for (lef_f = 0; lef_f < 12; lef_f = lef_f + 1) begin : LEF1_FILTER
            for (lef_d = 0; lef_d < 32; lef_d = lef_d + 1) begin : LEF1_DEPTH
                for (lef_t = 0; lef_t < 9; lef_t = lef_t + 1) begin : LEF1_TAP
                    assign P_lef1[lef_f*32 + lef_d][lef_t] = { {DATA_WIDTH-1{1'b0}}, conv_in[lef_d][lef_f][lef_t] };
                end
            end
        end
    endgenerate

    // always @(posedge clk or negedge arst_n) begin
    //     if (!arst_n)
    //         frame_counter <= 0;
    //     else if (enable)
    //         frame_counter <= frame_counter + 1;
    // end

    // =========================================================
    // STAGE 2: 12x adder_tree_10_4_1_1
    //   Each takes 32 consecutive cov9 outputs (384/12 = 32)
    //   Produces: conv25[k][0:9] and final_output[k]
    // =========================================================
    wire signed [DATA_WIDTH-1:0] adder_conv25       [0:11][0:9];
    wire signed [DATA_WIDTH-1:0] adder_final_output [0:11];

    genvar k;
    generate
        for (k = 0; k < 12; k = k + 1) begin : ADDER_TREE_INST
            adder_tree_10_4_1_1 u_adder_tree (
                .in           (cov9_to_adder[k*(32) +: 31]),
                .conv25       (adder_conv25[k]),
                .final_output (adder_final_output[k])
            );
        end
    endgenerate

    // =========================================================
    // STAGE 2→3 BRIDGE: Prepare 4-input shaban units.
    //   src_sel == 0 : all 32 units use adder_conv25 inputs via adders.
    //   src_sel == 1 : units 0..2 use adder_final_output, others use adder_conv25.
    //   src_sel == 2 : unit 0 uses adder_final_output, others use adder_conv25.
    // =========================================================

    wire signed [DATA_WIDTH-1:0] shaban_conv_in [0:31][0:3];
    wire signed [DATA_WIDTH-1:0] shaban_default_conv_in [0:31][0:3];
    wire signed [DATA_WIDTH:0]   shaban_default_accum [0:31][0:3];
    wire                       shaban_spike  [0:31];

    genvar s, t;
    generate
        for (s = 0; s < 32; s = s + 1) begin : SHABAN_DEFAULT_ASSIGN
            assign shaban_default_accum[s][0] = adder_conv25[s % 12][0] + adder_conv25[s % 12][1];
            assign shaban_default_accum[s][1] = adder_conv25[s % 12][2] + adder_conv25[s % 12][3];
            assign shaban_default_accum[s][2] = adder_conv25[s % 12][4] + adder_conv25[s % 12][5];
            assign shaban_default_accum[s][3] = adder_conv25[s % 12][6] + adder_conv25[s % 12][7];

            for (t = 0; t < 4; t = t + 1) begin : SHABAN_DEFAULT_TAP
                assign shaban_default_conv_in[s][t] = shaban_default_accum[s][t][DATA_WIDTH-1:0];
            end
        end

        for (s = 0; s < 32; s = s + 1) begin : SHABAN_INPUT_ASSIGN
            for (t = 0; t < 4; t = t + 1) begin : SHABAN_INPUT_TAP
                assign shaban_conv_in[s][t] =
                    (src_sel == 2'b10 && s == 0 && t == 0) ? adder_final_output[0] :
                    (src_sel == 2'b10 && s == 0) ? '0 :
                    (src_sel == 2'b01 && s < 3) ? adder_final_output[s*4 + t] :
                    shaban_default_conv_in[s][t];
            end
        end
    endgenerate

    // =========================================================
    // STAGE 3: 32x shaban_unit_top
    //   Each unit receives 4 inputs.
    // =========================================================
    genvar m;
    generate
        for (m = 0; m < 32; m = m + 1) begin : SHABAN_INST
            shaban_unit_top #(
                .DATA_WIDTH         (DATA_WIDTH),
                .conv_bias_relu_num (4),
                .batch_norm_num     (4),
                .pool_num           (2)
            ) u_shaban (
                .clk        (clk),
                .rst        (rst),
                .conv_in    ({ shaban_conv_in[m][3], shaban_conv_in[m][2],
                               shaban_conv_in[m][1], shaban_conv_in[m][0] }),
                .conv_bias  (conv_bias),
                .mult_wight (mult_wight),
                .add_wight  (add_wight),
                .spike      (shaban_spike[m])
            );
        end
    endgenerate

    assign spike[0] = shaban_spike[0];
    assign spike[1] = shaban_spike[1];
    assign spike[2] = shaban_spike[2];

endmodule
