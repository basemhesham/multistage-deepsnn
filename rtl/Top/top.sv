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
    parameter int CLASSIFIER_FRAC_BITS = 9,
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
    output logic [(4*DATA_WIDTH)-1:0]      class_logits,
    output logic                          classifier_done,
    output logic                          classifier_busy,
    output logic                          snn_done,
    output logic                          done

);

    localparam int FC1_INPUTS  = 128;
    localparam int FC1_OUTPUTS = 256;
    localparam int FC2_OUTPUTS = 4;

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
    logic                          gap_sample_enable;
    logic                          gap_clear;
    logic                          gap_done;
    logic                          gap_busy;
    logic                          enable_d;
    logic                          snn_done_d;
    logic                          spike_mem_wr_en;
    logic [3199:0]                 spike_mem_wr_data;
    logic [(384*PIXEL_W)-1:0]      pixel_mem_data;
    logic [3199:0]                 spike_mem_data;
    logic signed [DATA_WIDTH-1:0]  conv_bias_param [0:N_SHAABAN-1];
    logic signed [DATA_WIDTH-1:0]  mult_weight_param [0:N_SHAABAN-1];
    logic signed [DATA_WIDTH-1:0]  add_weight_param  [0:N_SHAABAN-1];
    logic signed [DATA_WIDTH-1:0]  fc1_in  [0:FC1_INPUTS-1];
    logic signed [DATA_WIDTH-1:0]  fc1_out [0:FC1_OUTPUTS-1];
    logic signed [DATA_WIDTH-1:0]  fc2_out [0:FC2_OUTPUTS-1];
    logic                          fc1_start;
    logic                          fc1_done;
    logic                          fc1_busy;
    logic                          fc2_start;
    logic                          fc2_done;
    logic                          fc2_busy;

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
        .gap_valid      (gap_sample_enable),
        .done           (snn_done)
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

    logic signed [PIXEL_W-1:0] pixels_mapped [0:11][0:31][0:8];
    logic signed [PIXEL_W-1:0] weights_mapped [0:11][0:31][0:8];
    logic signed [DATA_WIDTH-1:0] mac_to_connect [0:11][0:31];
    logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_bus [0:N_SHAABAN-1];
    logic [0:31] shaaban_spike_bus [0:31];

    assign spike_mem_wr_en = enable && (|ctrl_mem_enable);

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

    top_pixel_source_mapper #(
        .PIXEL_W        (PIXEL_W),
        .FRAME_NO       (FRAME_NO),
        .FRAME_NO_WIDTH (FRAME_NO_WIDTH)
    ) u_pixel_source_mapper (
        .clk            (clk),
        .arst_n         (arst_n),
        .src_sel        (src_sel),
        .frame          (frame),
        .pixel_mem_data (pixel_mem_data),
        .spike_mem_data (spike_mem_data),
        .pixels_mapped  (pixels_mapped)
    );

    top_weight_mapper #(
        .PIXEL_W (PIXEL_W)
    ) u_weight_mapper (
        .src_sel        (src_sel),
        .conv2_filter   (conv2_filter),
        .conv3_filter   (conv3_filter),
        .weights_mapped (weights_mapped)
    );

    top_conv9_array #(
        .PIXEL_W    (PIXEL_W),
        .MAC_OUT_W  (MAC_OUT_W),
        .DATA_WIDTH (DATA_WIDTH)
    ) u_conv9_array (
        .clk            (clk),
        .pixels_mapped  (pixels_mapped),
        .weights_mapped (weights_mapped),
        .mac_to_connect (mac_to_connect)
    );

    adder_tree_shaaban_connect u_connect (
        .clk          (clk),
        .rst          (rst),
        .src_sel      (src_sel),        // stage selector -- routes MUX inside
        .mac_in       (mac_to_connect), // 12x32 truncated conv9 outputs
        .shb_conv_bus (shb_bus)         // 32 packed Shaaban input buses
    );

    top_shaaban_array #(
        .DATA_WIDTH     (DATA_WIDTH),
        .N_SHAABAN      (N_SHAABAN),
        .INPUTS_PER_SHB (INPUTS_PER_SHB)
    ) u_shaaban_array (
        .clk                 (clk),
        .rst                 (rst),
        .shb_bus             (shb_bus),
        .conv_bias_param     (conv_bias_param),
        .mult_weight_param   (mult_weight_param),
        .add_weight_param    (add_weight_param),
        .spike_out           (spike_out),
        .shaaban_spike_bus   (shaaban_spike_bus)
    );

    top_spike_writeback u_spike_writeback (
        .stage_sel           (stage_sel),
        .shaaban_spike_bus   (shaaban_spike_bus),
        .spike_mem_wr_data   (spike_mem_wr_data)
    );

    always_ff @(posedge clk or negedge arst_n) begin
        if (!arst_n) begin
            enable_d   <= 1'b0;
            snn_done_d <= 1'b0;
        end else if (rst) begin
            enable_d   <= 1'b0;
            snn_done_d <= 1'b0;
        end else begin
            enable_d   <= enable;
            snn_done_d <= snn_done;
        end
    end

    assign gap_clear       = enable && !enable_d;
    assign fc1_start       = gap_done;
    assign fc2_start       = fc1_done;
    assign classifier_done = fc2_done;
    assign classifier_busy = gap_busy || fc1_busy || fc2_busy;
    assign done            = classifier_done;

    global_average_pool #(
        .DATA_WIDTH   (DATA_WIDTH),
        .FRAC_BITS    (CLASSIFIER_FRAC_BITS),
        .CHANNELS     (FC1_INPUTS),
        .SAMPLE_COUNT (CTRL_FRAGMENTS_MAX)
    ) u_global_average_pool (
        .clk            (clk),
        .rst            (rst),
        .clear          (gap_clear),
        .sample_valid   ((src_sel == 2'b10) && gap_sample_enable && ctrl_mem_enable[conv3_filter]),
        .sample_channel (conv3_filter),
        .sample_spike   (spike_out[0]),
        .start          (snn_done && !snn_done_d),
        .pool_out       (fc1_in),
        .done           (gap_done),
        .busy           (gap_busy)
    );

    fc1_layer #(
        .DATA_WIDTH  (DATA_WIDTH),
        .FRAC_BITS   (CLASSIFIER_FRAC_BITS),
        .N_INPUTS    (FC1_INPUTS),
        .N_OUTPUTS   (FC1_OUTPUTS)
    ) u_fc1_layer (
        .clk    (clk),
        .rst    (rst),
        .fc_in  (fc1_in),
        .start  (fc1_start),
        .fc_out (fc1_out),
        .done   (fc1_done),
        .busy   (fc1_busy)
    );

    fc2_layer #(
        .DATA_WIDTH (DATA_WIDTH),
        .FRAC_BITS  (CLASSIFIER_FRAC_BITS),
        .N_INPUTS   (FC1_OUTPUTS),
        .N_OUTPUTS  (FC2_OUTPUTS)
    ) u_fc2_layer (
        .clk    (clk),
        .rst    (rst),
        .fc_in  (fc1_out),
        .start  (fc2_start),
        .fc_out (fc2_out),
        .done   (fc2_done),
        .busy   (fc2_busy)
    );

    top_class_logits_packer #(
        .DATA_WIDTH (DATA_WIDTH),
        .N_OUTPUTS  (FC2_OUTPUTS)
    ) u_class_logits_packer (
        .fc_out       (fc2_out),
        .class_logits (class_logits)
    );

endmodule
