//////////////////////////////////////////////////////////////////////////////////////////////////
// Author    : Ahmad Khattab
// Date      : 5/13/26
// File      : fc1_layer.sv
// Status    : finalized
// Goal      : Fully-connected layer 1: 128 inputs -> 256 outputs + ReLU.
//             Uses 8 parallel MAC units with time-multiplexed dot-product
//             computation across 32 batches. Q7.10 fixed-point throughout.
//             Weights included from FC1_WEIGHTS.sv, biases defined inline.
//////////////////////////////////////////////////////////////////////////////////////////////////

module fc1_layer #(
    parameter int DATA_WIDTH     = 18,
    parameter int FRAC_BITS      = 10,
    parameter int N_INPUTS       = 128,
    parameter int N_OUTPUTS      = 256,
    parameter int ACCUM_WIDTH    = 48,
    parameter int PARALLEL_MACS  = 8
)(
    input  logic                           clk,
    input  logic                           rst,

    input  logic signed [DATA_WIDTH-1:0]   fc_in  [0:N_INPUTS-1],
    input  logic                           start,

    output logic signed [DATA_WIDTH-1:0]   fc_out [0:N_OUTPUTS-1],
    output logic                           done,
    output logic                           busy
);

    `include "../SNN Parameters/Values For LUTS Before Optimization/FC1_WEIGHTS.sv"

    localparam logic signed [DATA_WIDTH-1:0] FC1_BIAS [N_OUTPUTS] = '{
        18'h3FFE7, 18'h00000, 18'h0001F, 18'h0001B, 18'h00007, 18'h3FFFF, 18'h0001F,
        18'h00000, 18'h00000, 18'h00019, 18'h0001F, 18'h3FFFF, 18'h00013, 18'h3FFDB,
        18'h00040, 18'h00000, 18'h0002D, 18'h00000, 18'h00005, 18'h00002, 18'h3FFF6,
        18'h3FFF5, 18'h00000, 18'h00026, 18'h00016, 18'h3FFE8, 18'h00000, 18'h00000,
        18'h00000, 18'h0000C, 18'h00011, 18'h00002, 18'h3FFED, 18'h00000, 18'h3FFF7,
        18'h00000, 18'h3FFED, 18'h3FFEC, 18'h00018, 18'h3FFFE, 18'h0000F, 18'h00000,
        18'h0002A, 18'h00000, 18'h3FFFB, 18'h3FFFF, 18'h3FFE9, 18'h00024, 18'h3FFE8,
        18'h00000, 18'h00014, 18'h3FFE9, 18'h3FFE9, 18'h3FFDC, 18'h00002, 18'h00000,
        18'h3FFF0, 18'h00018, 18'h3FFFA, 18'h00000, 18'h3FFFA, 18'h00009, 18'h00000,
        18'h00000, 18'h00010, 18'h00018, 18'h00000, 18'h3FFE8, 18'h00001, 18'h00000,
        18'h3FFE0, 18'h00022, 18'h3FFF7, 18'h00035, 18'h00000, 18'h00000, 18'h3FFEE,
        18'h00000, 18'h00000, 18'h0002F, 18'h3FFEB, 18'h3FFFA, 18'h00000, 18'h00012,
        18'h3FFDF, 18'h3FFFF, 18'h00000, 18'h3FFE2, 18'h00004, 18'h00000, 18'h0002B,
        18'h0000F, 18'h00019, 18'h0001A, 18'h00013, 18'h00000, 18'h00016, 18'h00000,
        18'h00041, 18'h3FFED, 18'h3FFFC, 18'h00000, 18'h00000, 18'h3FFFB, 18'h3FFE6,
        18'h0001B, 18'h3FFFD, 18'h0001A, 18'h3FFFC, 18'h00000, 18'h3FFD5, 18'h3FFEA,
        18'h00000, 18'h3FFE1, 18'h00006, 18'h00000, 18'h3FFEB, 18'h3FFED, 18'h00022,
        18'h0000D, 18'h00009, 18'h3FFEA, 18'h3FFE7, 18'h3FFE9, 18'h00002, 18'h00000,
        18'h0000C, 18'h3FFF6, 18'h3FFFF, 18'h3FFE9, 18'h3FFFD, 18'h00006, 18'h00018,
        18'h00000, 18'h3FFF4, 18'h0000A, 18'h3FFF3, 18'h00007, 18'h3FFFF, 18'h00000,
        18'h00000, 18'h3FFE3, 18'h3FFF6, 18'h3FFFC, 18'h3FFF6, 18'h00000, 18'h00027,
        18'h00003, 18'h0000A, 18'h0000B, 18'h00001, 18'h00000, 18'h00000, 18'h00008,
        18'h3FFDC, 18'h3FFF6, 18'h00000, 18'h00022, 18'h00000, 18'h00023, 18'h3FFEA,
        18'h3FFE0, 18'h3FFE1, 18'h00012, 18'h3FFF7, 18'h0001B, 18'h3FFE7, 18'h00017,
        18'h00000, 18'h00000, 18'h00007, 18'h00014, 18'h3FFE0, 18'h0000B, 18'h3FFFA,
        18'h3FFFE, 18'h3FFF4, 18'h00000, 18'h0000C, 18'h0000C, 18'h0001E, 18'h00014,
        18'h0001C, 18'h3FFF4, 18'h00000, 18'h3FFE5, 18'h3FFFC, 18'h00031, 18'h3FFE2,
        18'h3FFDC, 18'h00034, 18'h3FFE5, 18'h0000A, 18'h3FFFF, 18'h00000, 18'h3FFFD,
        18'h3FFF6, 18'h00027, 18'h00000, 18'h00010, 18'h00017, 18'h0000C, 18'h00000,
        18'h3FFF1, 18'h00000, 18'h3FFF1, 18'h3FFFD, 18'h00000, 18'h3FFFE, 18'h3FFDB,
        18'h0001E, 18'h00005, 18'h3FFFC, 18'h3FFFC, 18'h3FFFD, 18'h00000, 18'h00010,
        18'h3FFCC, 18'h0001A, 18'h3FFDF, 18'h00016, 18'h00006, 18'h3FFED, 18'h00000,
        18'h3FFCD, 18'h00000, 18'h00028, 18'h3FFFF, 18'h0001C, 18'h00017, 18'h3FFEE,
        18'h0000A, 18'h0000B, 18'h0000A, 18'h0000D, 18'h3FFE5, 18'h0001B, 18'h00035,
        18'h00003, 18'h00007, 18'h0002C, 18'h00000, 18'h0001F, 18'h0000D, 18'h0000A,
        18'h3FFE1, 18'h00015, 18'h00000, 18'h0000F, 18'h0002A, 18'h3FFFF, 18'h00002,
        18'h00006, 18'h00005, 18'h00000, 18'h3FFF6
    };

    localparam int NUM_BATCHES  = N_OUTPUTS / PARALLEL_MACS;
    localparam int BATCH_BITS   = $clog2(NUM_BATCHES);
    localparam int INPUT_BITS   = $clog2(N_INPUTS);

    typedef enum logic [1:0] {
        ST_IDLE, ST_COMPUTE, ST_DONE
    } state_t;
    state_t state, next_state;

    logic [BATCH_BITS-1:0]   batch_idx;
    logic [INPUT_BITS-1:0]   input_idx;
    logic                    batch_done;
    logic                    compute_active;
    logic [$clog2(N_OUTPUTS)-1:0] completed_batch_base;

    logic signed [ACCUM_WIDTH-1:0] acc_mac [0:PARALLEL_MACS-1];

    logic signed [DATA_WIDTH-1:0]   weight [0:PARALLEL_MACS-1];
    logic signed [DATA_WIDTH-1:0]   in_val;
    logic signed [35:0]             product [0:PARALLEL_MACS-1];

    logic [$clog2(N_OUTPUTS)-1:0]   neuron_base;
    assign neuron_base = batch_idx * PARALLEL_MACS;
    assign in_val = fc_in[input_idx];

    // Weight ROM read — combinational
    generate
        for (genvar m = 0; m < PARALLEL_MACS; m++) begin : gen_weight_rd
            assign weight[m] = FC1_WEIGHTS[neuron_base + m][input_idx];
        end
    endgenerate

    // Products — synthesizer maps to DSP48E2
    generate
        for (genvar m = 0; m < PARALLEL_MACS; m++) begin : gen_product
            assign product[m] = in_val * weight[m];
        end
    endgenerate

    // FSM state register
    always_ff @(posedge clk) begin
        if (rst)
            state <= ST_IDLE;
        else
            state <= next_state;
    end

    // FSM next-state logic
    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE:    if (start)                                       next_state = ST_COMPUTE;
            ST_COMPUTE: if (batch_done && batch_idx == NUM_BATCHES - 1)  next_state = ST_DONE;
            ST_DONE:                                                     next_state = ST_IDLE;
            default:                                                     next_state = ST_IDLE;
        endcase
    end

    // Datapath control
    always_ff @(posedge clk) begin
        if (rst) begin
            batch_idx            <= '0;
            input_idx            <= '0;
            batch_done           <= 1'b0;
            compute_active       <= 1'b0;
            completed_batch_base <= '0;
            busy                 <= 1'b0;
            done                 <= 1'b0;
        end else begin
            done       <= 1'b0;
            batch_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        batch_idx      <= '0;
                        input_idx      <= '0;
                        compute_active <= 1'b1;
                        busy           <= 1'b1;
                    end
                end

                ST_COMPUTE: begin
                    busy <= 1'b1;
                    if (compute_active) begin
                        if (input_idx == N_INPUTS - 1) begin
                            completed_batch_base <= batch_idx * PARALLEL_MACS;
                            input_idx <= '0;
                            if (batch_idx == NUM_BATCHES - 1) begin
                                compute_active <= 1'b0;
                                batch_done     <= 1'b1;
                            end else begin
                                batch_idx  <= batch_idx + 1'b1;
                                batch_done <= 1'b1;
                            end
                        end else begin
                            input_idx <= input_idx + 1'b1;
                        end
                    end
                end

                ST_DONE: begin
                    done <= 1'b1;
                    busy <= 1'b0;
                end

                default: ;
            endcase
        end
    end

    // MAC accumulators — reset at batch start, accumulate each cycle
    always_ff @(posedge clk) begin
        if (rst)
            for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= '0;
        else if (state == ST_IDLE && start)
            for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= '0;
        else if (state == ST_COMPUTE && compute_active) begin
            if (input_idx == 0)
                for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= $signed(product[m]);
            else
                for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= acc_mac[m] + $signed(product[m]);
        end
    end

    // Batch complete — apply bias, normalise (>>>FRAC_BITS), ReLU, store
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int o = 0; o < N_OUTPUTS; o++) fc_out[o] <= '0;
        end else if (state == ST_COMPUTE && batch_done) begin
            for (int m = 0; m < PARALLEL_MACS; m++) begin
                automatic logic signed [ACCUM_WIDTH-1:0] bias_wide, bias_aligned, total;
                automatic logic signed [DATA_WIDTH-1:0]  result;
                automatic int unsigned neuron;
                neuron       = completed_batch_base + m;
                bias_wide    = $signed(FC1_BIAS[neuron]);
                bias_aligned = bias_wide <<< FRAC_BITS;
                total        = acc_mac[m] + bias_aligned;
                result       = total >>> FRAC_BITS;
                if (result < 0) result = '0;
                fc_out[neuron] <= result;
            end
        end
    end

endmodule
