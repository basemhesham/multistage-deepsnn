`timescale 1ns / 1ps

module spike_mem #(
    parameter int MEM_WORD   = 3200,
    parameter int ADDR_WIDTH = 6
)(
    input  logic                  clk,
    input  logic                  rst,
    input  logic                  wr_en,
    input  logic [0:MEM_WORD-1]   bit_enable,
    input  logic                  zero_sel,
    input  logic [ADDR_WIDTH-1:0] wr_addr,
    input  logic [ADDR_WIDTH-1:0] rd_addr,
    input  logic [MEM_WORD-1:0]   wr_data,
    output logic [MEM_WORD-1:0]   rd_data
);

    localparam int DEPTH = 1 << ADDR_WIDTH;

    (* ram_style = "block" *)
    logic [MEM_WORD-1:0] mem [0:DEPTH-1];

    logic [MEM_WORD-1:0] write_shadow;
    logic [MEM_WORD-1:0] merge_base;
    logic [MEM_WORD-1:0] next_write_word;
    logic [ADDR_WIDTH-1:0] active_wr_addr;
    logic active_wr_addr_valid;
    logic same_write_word;

    integer merge_idx;

    assign same_write_word = active_wr_addr_valid && (active_wr_addr == wr_addr);

    always_comb begin
        merge_base      = same_write_word ? write_shadow : '0;
        next_write_word = merge_base;

        for (merge_idx = 0; merge_idx < MEM_WORD; merge_idx = merge_idx + 1) begin
            if (bit_enable[merge_idx]) begin
                next_write_word[merge_idx] = zero_sel ? 1'b0 : wr_data[merge_idx];
            end
        end
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            rd_data              <= '0;
            write_shadow         <= '0;
            active_wr_addr       <= '0;
            active_wr_addr_valid <= 1'b0;
        end else begin
            if (wr_en) begin
                mem[wr_addr]          <= next_write_word;
                write_shadow         <= next_write_word;
                active_wr_addr       <= wr_addr;
                active_wr_addr_valid <= 1'b1;
            end

            rd_data <= mem[rd_addr];
        end
    end

endmodule
