`timescale 1ns / 1ps

module top_spike_writeback (
    input  logic [1:0]  stage,
    input  logic [0:31] shaaban_spike_bus [0:31],
    output logic [3199:0] spike_mem_wr_data
);

    logic [0:31] mem_mapped_internal [0:3199];

    mem_maping_1_2 u_writeback (
        .stage_sel   (stage == 2'b00 ? 1'b0 : 1'b1),
        .shaaban_out (shaaban_spike_bus),
        .mem_mapped  (mem_mapped_internal)
    );

    genvar wb;
    generate
        for (wb = 0; wb < 3200; wb++) begin : gen_spike_mem_wr_data
            always_comb begin
                if (stage == 2'b10)
                    spike_mem_wr_data[wb] = shaaban_spike_bus[0][0];
                else
                    spike_mem_wr_data[wb] = mem_mapped_internal[wb][0];
            end
        end
    endgenerate

endmodule
