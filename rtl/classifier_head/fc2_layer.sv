//////////////////////////////////////////////////////////////////////////////////////////////////
// Author    : Ahmad Khattab
// Date      : 5/13/26
// File      : fc2_layer.sv
// Status    : finalized
// Goal      : Fully-connected layer 2: 256 inputs -> 4 outputs (class logits, no ReLU).
//             All 4 output neurons computed in parallel with single-batch MAC array.
//             Q7.10 fixed-point. Weights included from FC2_WEIGHTS.sv, biases inline.
//////////////////////////////////////////////////////////////////////////////////////////////////

module fc2_layer #(
    parameter int DATA_WIDTH     = 18,
    parameter int FRAC_BITS      = 10,
    parameter int N_INPUTS       = 256,
    parameter int N_OUTPUTS      = 4,
    parameter int ACCUM_WIDTH    = 48
)(
    input  logic                           clk,
    input  logic                           rst,

    input  logic signed [DATA_WIDTH-1:0]   fc_in  [0:N_INPUTS-1],
    input  logic                           start,

    output logic signed [DATA_WIDTH-1:0]   fc_out [0:N_OUTPUTS-1],
    output logic                           done,
    output logic                           busy
);

    `include "../SNN Parameters/Values For LUTS Before Optimization/FC2_WEIGHTS.sv"

    localparam logic signed [DATA_WIDTH-1:0] FC2_BIAS [N_OUTPUTS] = '{
        18'h00011, 18'h3FFDD, 18'h3FF69, 18'h0009A
    };

    localparam int INPUT_BITS    = $clog2(N_INPUTS);
    localparam int PARALLEL_MACS = N_OUTPUTS;

    typedef enum logic [1:0] {
        ST_IDLE, ST_COMPUTE, ST_DONE
    } state_t;
    state_t state, next_state;

    logic [INPUT_BITS-1:0]   input_idx;
    logic                    compute_done;

    logic signed [ACCUM_WIDTH-1:0] acc_mac [0:PARALLEL_MACS-1];

    logic signed [DATA_WIDTH-1:0]   weight [0:PARALLEL_MACS-1];
    logic signed [DATA_WIDTH-1:0]   in_val;
    logic signed [35:0]             product [0:PARALLEL_MACS-1];

    assign in_val = fc_in[input_idx];

    // Weight ROM read
    generate
        for (genvar m = 0; m < PARALLEL_MACS; m++) begin : gen_weight_rd
            assign weight[m] = FC2_WEIGHTS[m][input_idx];
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
        if (rst) state <= ST_IDLE;
        else     state <= next_state;
    end

    // FSM next-state
    always_comb begin
        next_state = state;
        case (state)
            ST_IDLE:    if (start)         next_state = ST_COMPUTE;
            ST_COMPUTE: if (compute_done)  next_state = ST_DONE;
            ST_DONE:                       next_state = ST_IDLE;
            default:                       next_state = ST_IDLE;
        endcase
    end

    // Datapath control
    always_ff @(posedge clk) begin
        if (rst) begin
            input_idx    <= '0;
            compute_done <= 1'b0;
            busy         <= 1'b0;
            done         <= 1'b0;
        end else begin
            done         <= 1'b0;
            compute_done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin input_idx <= '0; busy <= 1'b1; end
                end

                ST_COMPUTE: begin
                    busy <= 1'b1;
                    if (input_idx == N_INPUTS - 1) begin
                        input_idx    <= '0;
                        compute_done <= 1'b1;
                    end else begin
                        input_idx <= input_idx + 1'b1;
                    end
                end

                ST_DONE: begin done <= 1'b1; busy <= 1'b0; end
                default: ;
            endcase
        end
    end

    // MAC accumulators
    always_ff @(posedge clk) begin
        if (rst)
            for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= '0;
        else if (state == ST_IDLE && start)
            for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= '0;
        else if (state == ST_COMPUTE) begin
            if (input_idx == 0)
                for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= $signed(product[m]);
            else
                for (int m = 0; m < PARALLEL_MACS; m++) acc_mac[m] <= acc_mac[m] + $signed(product[m]);
        end
    end

    // Completion — bias add, normalise (>>>FRAC_BITS), store (no ReLU)
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int o = 0; o < N_OUTPUTS; o++) fc_out[o] <= '0;
        end else if (state == ST_COMPUTE && compute_done) begin
            for (int m = 0; m < PARALLEL_MACS; m++) begin
                automatic logic signed [ACCUM_WIDTH-1:0] bias_wide, bias_aligned, total;
                automatic logic signed [DATA_WIDTH-1:0]  result;
                bias_wide    = $signed(FC2_BIAS[m]);
                bias_aligned = bias_wide <<< FRAC_BITS;
                total        = acc_mac[m] + bias_aligned;
                result       = total >>> FRAC_BITS;
                fc_out[m] <= result;
            end
        end
    end

endmodule
