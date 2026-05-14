module deep_snn_top #(
    parameter int PIXEL_W        = 18,
    parameter int MAC_OUT_W      = 40,
    parameter int DATA_WIDTH     = 18,
    parameter int N_SHAABAN      = 32,
    parameter int INPUTS_PER_SHB = 4
)(
    input  logic                    clk,
    input  logic                    rst,

    // Control Signals
    input  logic [1:0]              src_sel,
    input  wire                     arst_n,
    input  wire                     enable,

    // Memory Interface (for loading pixels and weights, if needed)
    input  wire [3199:0]            mem,
    

    // Inputs (24x24 crop pixels and weights) - single 9-tap vectors.
    // Convolution input/output mapping block to be inserted here later.
    input  logic signed  [PIXEL_W-1:0]      pixels  [0:8],
    input  logic signed  [PIXEL_W-1:0]      weights [0:8],

    // Shaaban Hyperparameters
    input  logic signed [DATA_WIDTH-1:0] conv_bias,
    input  logic signed [DATA_WIDTH-1:0] mult_weight,
    input  logic signed [DATA_WIDTH-1:0] add_weight,

    output logic [N_SHAABAN-1:0]    spike_out
);

    // -------------------------------------------------------------------------
    // Interconnects
    // -------------------------------------------------------------------------
    logic [MAC_OUT_W-1:0]                          mac_raw        [0:11][0:31];
    logic signed [11:0][31:0][DATA_WIDTH-1:0]      mac_to_connect;
    logic signed [(INPUTS_PER_SHB*DATA_WIDTH)-1:0] shb_bus        [0:N_SHAABAN-1];
    
    logic signed [PIXEL_W-1:0] in_mem [0:383];   // flat pixel input
    logic signed [PIXEL_W-1:0] p_imag [0:383];   // remapped output → feeds conv9

    // -------------------------------------------------------------------------
    // 1. Convolution Array - 384 parallel conv9 units (12 rows x 32 cols).
    //    All units currently share the same pixels/weights inputs.
    //    TODO: replace broadcast connections below with the mapping block.
    // -------------------------------------------------------------------------
    genvar g, c;
    for (g = 0; g < 12; g++) begin : gen_conv_row
        for (c = 0; c < 32; c++) begin : gen_conv_col
            conv9 #(
                .PIXEL_W (PIXEL_W),
                .PROD_W  (36),
                .OUT_W   (MAC_OUT_W)
            ) u_conv (
                .P         (pixels),   // TODO: remap per mapping block
                .Q         (weights),  // TODO: remap per mapping block
                .Pixel_Out (mac_raw[g][c])
            );
        end
    end

    // -------------------------------------------------------------------------
    // 2. Truncation / Fixed-point normalisation
    //    Keep the DATA_WIDTH LSBs of each 40-bit MAC result.
    // -------------------------------------------------------------------------
    genvar g2, c2;
    for (g2 = 0; g2 < 12; g2++) begin : gen_trunc_row
        for (c2 = 0; c2 < 32; c2++) begin : gen_trunc_col
            assign mac_to_connect[g2][c2] = mac_raw[g2][c2][DATA_WIDTH-1:0];
        end
    end

    // -------------------------------------------------------------------------
    // 3. Summation Engine (Stages 1, 2, 3 + Routing)
    // -------------------------------------------------------------------------
    adder_tree_shaaban_connect u_connect (
        .clk         (clk),
        .rst         (rst),
        .src_sel     (src_sel),
        .mac_in      (mac_to_connect),
        .shb_conv_bus(shb_bus)
    );

    // -------------------------------------------------------------------------
    // 4. Shaaban Processing Array (32 Units)
    // -------------------------------------------------------------------------
    genvar s;
    for (s = 0; s < N_SHAABAN; s++) begin : gen_shaaban_array
        shaban_unit_top #(
            .DATA_WIDTH         (DATA_WIDTH),
            .conv_bias_relu_num (INPUTS_PER_SHB),
            .batch_norm_num     (INPUTS_PER_SHB),
            .pool_num           (2)
        ) u_shb (
            .clk        (clk),
            .rst        (rst),
            .conv_in    (shb_bus[s]),
            .conv_bias  (conv_bias),
            .mult_wight (mult_weight),
            .add_wight  (add_weight),
            .spike      (spike_out[s])
        );
    end

endmodule