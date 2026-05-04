module deep_snn_top #(
    parameter int PIXEL_W        = 18,
    parameter int MAC_OUT_W      = 40,
    parameter int DATA_WIDTH     = 18,
    parameter int N_SHAABAN      = 32,
    parameter int INPUTS_PER_SHB = 4
)(
    input  logic                    clk,
    input  logic                    rst,
    input  logic [1:0]              src_sel,

    // Inputs (24x24 crop pixels and weights)
    input  logic [PIXEL_W-1:0]      pixels  [0:11][0:31][0:8],
    input  logic [PIXEL_W-1:0]      weights [0:11][0:31][0:8],

    // Shaaban Hyperparameters
    input  logic signed [DATA_WIDTH-1:0] conv_bias,
    input  logic signed [DATA_WIDTH-1:0] mult_weight,
    input  logic signed [DATA_WIDTH-1:0] add_weight,

    output logic [N_SHAABAN-1:0]    spike_out
);

    // Interconnects
    logic [MAC_OUT_W-1:0] mac_raw [0:11][0:31];
    logic signed [11:0][31:0][DATA_WIDTH-1:0] mac_to_connect;
    logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_bus [0:N_SHAABAN-1];

    // 1. Convolution Array (384 Parallel MACs)
    conv9_array u_conv (
        .pixels(pixels), .weights(weights), .mac_out(mac_raw)
    );

    // 2. Truncation/Fixed-point normalization (Adjust bit-range as needed)
    genvar g, c;
    generate
        for (g = 0; g < 12; g++) begin
            for (c = 0; c < 32; c++) begin
                assign mac_to_connect[g][c] = mac_raw[g][c][DATA_WIDTH-1:0];
            end
        end
    endgenerate

    // 3. Summation Engine (Stages 1, 2, 3 + Routing)
    adder_tree_shaaban_connect u_connect (
        .clk(clk), .rst(rst), .src_sel(src_sel),
        .mac_in(mac_to_connect),
        .shb_conv_bus(shb_bus)
    );

    // 4. Shaaban Processing Array (32 Units)
    genvar s;
    generate
        for (s = 0; s < N_SHAABAN; s++) begin : gen_shaaban_array
            shaban_unit_top #(
                .DATA_WIDTH(DATA_WIDTH),
                .conv_bias_relu_num(INPUTS_PER_SHB),
                .batch_norm_num(INPUTS_PER_SHB),
                .pool_num(2)
            ) u_shb (
                .clk(clk), .rst(rst),
                .conv_in(shb_bus[s]),
                .conv_bias(conv_bias),
                .mult_wight(mult_weight),
                .add_wight(add_weight),
                .spike(spike_out[s])
            );
        end
    endgenerate

endmodule