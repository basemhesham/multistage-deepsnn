// deep_snn_top integrates the current shared datapath.
// Architecture notes live in rtl/Top/README.md.

module deep_snn_top #(
    parameter int PIXEL_W        = 18,
    parameter int MAC_OUT_W      = 40,
    parameter int DATA_WIDTH     = 18,
    parameter int N_SHAABAN      = 32,
    parameter int INPUTS_PER_SHB = 4,
    parameter int FRAME_NO       = 6,
    parameter int FRAME_NO_WIDTH = 3,
    parameter int CTRL_FRAGMENT_ROWS   = 13,
    parameter int CTRL_FRAGMENT_COLS   = 13,
    parameter int CTRL_FRAGMENTS_MAX   = CTRL_FRAGMENT_ROWS * CTRL_FRAGMENT_COLS,
    parameter int CTRL_TEMPORAL_FRAMES = 16
)(
    input  logic                          clk,
    input  logic                          rst,
    input  wire                           arst_n,
    input  wire                           enable,

    input  logic                          pixel_mem_wr_en,
    input  logic [5:0]                    pixel_mem_wr_addr,
    input  logic [(384*PIXEL_W)-1:0]      pixel_mem_wr_data,

    output logic [N_SHAABAN-1:0]          spike_out,
    output logic                          done

);

    logic [1:0]                    src_sel;
    logic [FRAME_NO_WIDTH-1:0]     frame;
    logic                          stage_sel;
    logic [5:0]                    conv2_filter;
    logic [6:0]                    conv3_filter;
    logic [0:3199]                 ctrl_mem_enable;
    logic                          ctrl_rd_enable;
    logic [5:0]                    ctrl_rd_mem_adderss;
    logic [5:0]                    ctrl_wr_mem_adderss;
    logic                          ctrl_zero;
    logic                          ctrl_zero_sel;
    logic                          ctrl_padding_flag;
    logic                          spike_mem_wr_en;
    logic [3199:0]                 spike_mem_wr_data;
    logic [(384*PIXEL_W)-1:0]      pixel_mem_data;
    logic [3199:0]                 spike_mem_data;
    logic signed [DATA_WIDTH-1:0]  conv_bias_param [0:N_SHAABAN-1];
    logic signed [DATA_WIDTH-1:0]  mult_weight_param [0:N_SHAABAN-1];
    logic signed [DATA_WIDTH-1:0]  add_weight_param  [0:N_SHAABAN-1];

    top_controller #(
        .FRAGMENT_ROWS   (CTRL_FRAGMENT_ROWS),
        .FRAGMENT_COLS   (CTRL_FRAGMENT_COLS),
        .FRAGMENTS_MAX   (CTRL_FRAGMENTS_MAX),
        .TEMPORAL_FRAMES (CTRL_TEMPORAL_FRAMES)
    ) u_top_controller (
        .clk            (clk),
        .rst            (rst),
        .arst_n         (arst_n),
        .enable         (enable),
        .mem_enable     (ctrl_mem_enable),
        .rd_enable      (ctrl_rd_enable),
        .stage          (src_sel),
        .frame          (frame),
        .stage_sel      (stage_sel),
        .conv2_filter   (conv2_filter),
        .conv3_filter   (conv3_filter),
        .rd_mem_adderss (ctrl_rd_mem_adderss),
        .wr_mem_adderss (ctrl_wr_mem_adderss),
        .zero           (ctrl_zero),
        .zero_sel       (ctrl_zero_sel),
        .padding_flag   (ctrl_padding_flag),
        .done           (done)
    );

    pixel_mem #(
        .DATA_WIDTH  (PIXEL_W),
        .WORD_PIXELS (384),
        .ADDR_WIDTH  (6)
    ) u_pixel_mem (
        .clk     (clk),
        .rst     (rst),
        .wr_en   (pixel_mem_wr_en),
        .wr_addr (pixel_mem_wr_addr),
        .wr_data (pixel_mem_wr_data),
        .rd_addr (ctrl_rd_mem_adderss),
        .rd_data (pixel_mem_data)
    );

    bias_bn_params #(
        .DATA_WIDTH     (DATA_WIDTH),
        .N_SHAABAN      (N_SHAABAN),
        .CONV2_FILTER_W (6),
        .CONV3_FILTER_W (7)
    ) u_bias_bn_params (
        .stage       (src_sel),
        .conv2_filter(conv2_filter),
        .conv3_filter(conv3_filter),
        .conv_bias   (conv_bias_param),
        .mult_weight (mult_weight_param),
        .add_weight  (add_weight_param)
    );

    logic signed [PIXEL_W-1:0] in_mem [0:383];

    logic signed [PIXEL_W-1:0] p_imag [0:383];

    logic signed [PIXEL_W-1:0] pixels_s1 [0:11][0:31][0:8];

    logic fil_in [31:0][39:0];

    logic conv_windows [31:0][11:0][8:0];

    logic signed [PIXEL_W-1:0] pixels_s2 [0:11][0:31][0:8];

    logic stage3_mem [0:1023];

    logic stage3_windows [0:8][0:3][0:63];

    logic signed [PIXEL_W-1:0] pixels_s3 [0:11][0:31][0:8];

    logic signed [PIXEL_W-1:0] pixels_mapped [0:11][0:31][0:8];

    logic [PIXEL_W-1:0] stage1_weights [3456];   // 5x5 kernels, decomposed
    logic [PIXEL_W-1:0] stage2_weights [3456];   // 3x3 kernels, 32-channel
    logic [PIXEL_W-1:0] stage3_weights [3456];   // 3x3 kernels, 64-channel

    logic [PIXEL_W-1:0] active_weights [3456];

    logic signed [PIXEL_W-1:0] weights_mapped [0:11][0:31][0:8];

    logic [MAC_OUT_W-1:0] mac_raw [0:11][0:31];

    logic signed [DATA_WIDTH-1:0] mac_to_connect [0:11][0:31];

    logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_bus [0:N_SHAABAN-1];

    logic [0:31] shaaban_spike_bus [0:31];

    logic [0:31] mem_mapped_internal [0:3199];
    assign spike_mem_wr_en = enable && (|ctrl_mem_enable);

    genvar wb;
    generate
        for (wb = 0; wb < 3200; wb++) begin : gen_spike_mem_wr_data
            assign spike_mem_wr_data[wb] = mem_mapped_internal[wb][0];
        end
    endgenerate

    spike_mem #(
        .MEM_WORD   (3200),
        .ADDR_WIDTH (6)
    ) u_spike_mem (
        .clk        (clk),
        .rst        (rst),
        .wr_en      (spike_mem_wr_en),
        .bit_enable (ctrl_mem_enable),
        .zero_sel   (ctrl_zero_sel),
        .wr_addr    (ctrl_wr_mem_adderss),
        .rd_addr    (ctrl_rd_mem_adderss),
        .wr_data    (spike_mem_wr_data),
        .rd_data    (spike_mem_data)
    );

    genvar m;
    generate
        for (m = 0; m < 384; m++) begin : gen_inmem
            assign in_mem[m] = $signed(pixel_mem_data[m*PIXEL_W +: PIXEL_W]);
        end
    endgenerate

    genvar b, j;
    generate

        for (b = 0; b < 12; b = b + 1) begin : gen_state_0
            if (b % 3 == 0) begin : state_0
                for (j = 0; j < 32; j = j + 1) begin : assign_s0
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j];
                end
            end
        end

        for (b = 0; b < 12; b = b + 1) begin : gen_state_1
            if (b % 3 == 1) begin : state_1
                for (j = 0; j < 30; j = j + 1) begin : assign_s1
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 1];
                end
                assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];       // wrap: skipped element
                assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 31];  // seed: 1 orphan remains
            end
        end

        for (b = 0; b < 12; b = b + 1) begin : gen_state_2
            if (b % 3 == 2) begin : state_2
                for (j = 0; j < 30; j = j + 1) begin : assign_s2
                    assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 2];
                end
                assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];       // wrap: skipped[0]
                assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 1];   // wrap: skipped[1]
            end
        end

    endgenerate

    genvar gp1, cp1, tp1;
    generate
        for (gp1 = 0; gp1 < 12; gp1++) begin : gen_ps1_row
            for (cp1 = 0; cp1 < 32; cp1++) begin : gen_ps1_col
                for (tp1 = 0; tp1 < 9; tp1++) begin : gen_ps1_tap
                    assign pixels_s1[gp1][cp1][tp1] = p_imag[(gp1 * 32) + cp1];
                end
            end
        end
    endgenerate

    mem_mapping #(
        .FRAME_NO       (FRAME_NO),
        .FRAME_NO_WIDTH (FRAME_NO_WIDTH),
        .MEM_WORD       (3200)
    ) u_mem_mapping (
        .clk    (clk),
        .arst_n (arst_n),
        .frame  (frame),
        .mem    (spike_mem_data),
        .fil_in (fil_in)
    );

    genvar fi;
    generate
        for (fi = 0; fi < 32; fi++) begin : gen_frame_mapping
            frame_input_mapping u_frame_map (
                .frame (frame[2:0]),
                .in    (fil_in[fi]),
                .conv  (conv_windows[fi])
            );
        end
    endgenerate

    genvar gp2, cp2, tp2;
    generate
        for (gp2 = 0; gp2 < 12; gp2++) begin : gen_ps2_row
            for (cp2 = 0; cp2 < 32; cp2++) begin : gen_ps2_col
                for (tp2 = 0; tp2 < 9; tp2++) begin : gen_ps2_tap
                    assign pixels_s2[gp2][cp2][tp2] =
                        {{(PIXEL_W-1){conv_windows[cp2][gp2][tp2]}},  // sign replicate
                                      conv_windows[cp2][gp2][tp2]};   // LSB
                end
            end
        end
    endgenerate

    genvar sm;
    generate
        for (sm = 0; sm < 1024; sm++) begin : gen_stage3_mem_unpack
            assign stage3_mem[sm] = spike_mem_data[sm];
        end
    endgenerate

    bin_muxing_stage2 u_stage3_bin_mux (
        .din  (stage3_mem),
        .dout (stage3_windows)
    );

    genvar win3, ch3, tap3;
    generate
        for (win3 = 0; win3 < 4; win3++) begin : gen_ps3_window
            for (ch3 = 0; ch3 < 32; ch3++) begin : gen_ps3_channel
                for (tap3 = 0; tap3 < 9; tap3++) begin : gen_ps3_tap
                    assign pixels_s3[(win3 * 2)    ][ch3][tap3] =
                        {{(PIXEL_W-1){stage3_windows[tap3][win3][ch3]}},
                                      stage3_windows[tap3][win3][ch3]};

                    assign pixels_s3[(win3 * 2) + 1][ch3][tap3] =
                        {{(PIXEL_W-1){stage3_windows[tap3][win3][ch3 + 32]}},
                                      stage3_windows[tap3][win3][ch3 + 32]};
                end
            end
        end
    endgenerate

    genvar zrow3, zch3, ztap3;
    generate
        for (zrow3 = 8; zrow3 < 12; zrow3++) begin : gen_ps3_zero_row
            for (zch3 = 0; zch3 < 32; zch3++) begin : gen_ps3_zero_channel
                for (ztap3 = 0; ztap3 < 9; ztap3++) begin : gen_ps3_zero_tap
                    assign pixels_s3[zrow3][zch3][ztap3] = '0;
                end
            end
        end
    endgenerate

    genvar gm, cm, tm;
    generate
        for (gm = 0; gm < 12; gm++) begin : gen_pmux_row
            for (cm = 0; cm < 32; cm++) begin : gen_pmux_col
                for (tm = 0; tm < 9; tm++) begin : gen_pmux_tap
                    always_comb begin
                        case (src_sel)
                            2'b00:   pixels_mapped[gm][cm][tm] = pixels_s1[gm][cm][tm];  // Stage 1
                            2'b01:   pixels_mapped[gm][cm][tm] = pixels_s2[gm][cm][tm];  // Stage 2
                            2'b10:   pixels_mapped[gm][cm][tm] = pixels_s3[gm][cm][tm];  // Stage 3
                            default: pixels_mapped[gm][cm][tm] = pixels_s1[gm][cm][tm];
                        endcase
                    end
                end
            end
        end
    endgenerate

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
            2'b00:   active_weights = stage1_weights;   // Stage 1
            2'b01:   active_weights = stage2_weights;   // Stage 2
            2'b10:   active_weights = stage3_weights;   // Stage 3
            default: active_weights = stage2_weights;   // safe default
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

    genvar g, c;
    generate
        for (g = 0; g < 12; g++) begin : gen_conv_row
            for (c = 0; c < 32; c++) begin : gen_conv_col
                conv9 #(
                    .PIXEL_W (PIXEL_W),   // 18-bit Q7.10 signed inputs
                    .PROD_W  (36),        // 18x18 product width
                    .OUT_W   (MAC_OUT_W)  // 40-bit accumulator
                ) u_conv (
                    .CLK       (clk),
                    .P         (pixels_mapped[g][c]),   // 9 pixel taps
                    .Q         (weights_mapped[g][c]),  // 9 weight taps
                    .Pixel_Out (mac_raw[g][c])          // 40-bit MAC result
                );
            end
        end
    endgenerate

    genvar g2, c2;
    generate
        for (g2 = 0; g2 < 12; g2++) begin : gen_trunc_row
            for (c2 = 0; c2 < 32; c2++) begin : gen_trunc_col
                assign mac_to_connect[g2][c2] = mac_raw[g2][c2][DATA_WIDTH-1:0];
            end
        end
    endgenerate

    adder_tree_shaaban_connect u_connect (
        .clk          (clk),
        .rst          (rst),
        .src_sel      (src_sel),        // stage selector -- routes MUX inside
        .mac_in       (mac_to_connect), // 12x32 truncated conv9 outputs
        .shb_conv_bus (shb_bus)         // 32 packed Shaaban input buses
    );

    genvar s;
    generate
        for (s = 0; s < N_SHAABAN; s++) begin : gen_shaaban_array
            shaban_unit_top #(
                .DATA_WIDTH         (DATA_WIDTH),        // 18-bit data
                .conv_bias_relu_num (INPUTS_PER_SHB),    // 4 bias+relu units
                .batch_norm_num     (INPUTS_PER_SHB),    // 4 batch norm units
                .pool_num           (2)                  // 2:1 max pool stages
            ) u_shb (
                .clk        (clk),
                .rst        (rst),
                .conv_in    (shb_bus[s]),      // 4 packed 18-bit inputs
                .conv_bias  (conv_bias_param[s]),
                .mult_wight (mult_weight_param[s]),
                .add_wight  (add_weight_param[s]),
                .spike      (spike_out[s])     // 1-bit LIF output
            );

            assign shaaban_spike_bus[s] = {32{spike_out[s]}};
        end
    endgenerate

    mem_maping_1_2 u_writeback (
        .stage_sel   (stage_sel),           // layout selector (0=Stage1, 1=Stage2)
        .shaaban_out (shaaban_spike_bus),    // 32 x 32-bit spike words
        .mem_mapped  (mem_mapped_internal)  // 3200 x 32-bit output (internal only)
    );

endmodule
