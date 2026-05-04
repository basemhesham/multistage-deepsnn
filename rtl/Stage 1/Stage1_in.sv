genvar b, j;
generate

    // State 0: Straight Mapping (Blocks 0, 3, 6, 9)
    for (b = 0; b < 12; b = b + 1) begin : gen_state_0
        if (b % 3 == 0) begin : state_0
            for (j = 0; j < 32; j = j + 1) begin : assign_state_0
                assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j];
            end
        end
    end

    // State 1: 1-Shift Mapping (Blocks 1, 4, 7, 10)
    for (b = 0; b < 12; b = b + 1) begin : gen_state_1
        if (b % 3 == 1) begin : state_1
            for (j = 0; j < 30; j = j + 1) begin : assign_state_1
                assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 1];
            end
            assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];
            assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 31];
        end
    end

    // State 2: 2-Shift Mapping (Blocks 2, 5, 8, 11)
    for (b = 0; b < 12; b = b + 1) begin : gen_state_2
        if (b % 3 == 2) begin : state_2
            for (j = 0; j < 30; j = j + 1) begin : assign_state_2
                assign p_imag[(b * 32) + j] = in_mem[(b * 32) + j + 2];
            end
            assign p_imag[(b * 32) + 30] = in_mem[(b * 32)];
            assign p_imag[(b * 32) + 31] = in_mem[(b * 32) + 1];
        end
    end

endgenerate
