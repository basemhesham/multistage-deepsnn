`timescale 1ns / 1ps

module adder_tree_10_4_1_1_tb;

    reg  signed [17:0] in_1,  in_2,  in_3,  in_4,  in_5,  in_6,  in_7,  in_8;
    reg  signed [17:0] in_9,  in_10, in_11, in_12, in_13, in_14, in_15, in_16;
    reg  signed [17:0] in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24;
    reg  signed [17:0] in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32;

    wire signed [17:0] conv25_1, conv25_2, conv25_3, conv25_4, conv25_5;
    wire signed [17:0] conv25_6, conv25_7, conv25_8, conv25_9, conv25_10;
    wire signed [17:0] final_output;

    integer i;
    integer errors;
    integer seed;
    integer onehot_idx;
    reg signed [17:0] impulse_val;
    reg signed [47:0] exp_conv [0:9];
    reg signed [47:0] exp_total;
    localparam signed [17:0] MAX18 = 18'sd131071;
    localparam signed [17:0] MIN18 = -18'sd131072;

    adder_tree_10_4_1_1 dut (
        .in_1(in_1),   .in_2(in_2),   .in_3(in_3),   .in_4(in_4),
        .in_5(in_5),   .in_6(in_6),   .in_7(in_7),   .in_8(in_8),
        .in_9(in_9),   .in_10(in_10), .in_11(in_11), .in_12(in_12),
        .in_13(in_13), .in_14(in_14), .in_15(in_15), .in_16(in_16),
        .in_17(in_17), .in_18(in_18), .in_19(in_19), .in_20(in_20),
        .in_21(in_21), .in_22(in_22), .in_23(in_23), .in_24(in_24),
        .in_25(in_25), .in_26(in_26), .in_27(in_27), .in_28(in_28),
        .in_29(in_29), .in_30(in_30), .in_31(in_31), .in_32(in_32),
        .conv25_1(conv25_1), .conv25_2(conv25_2), .conv25_3(conv25_3), .conv25_4(conv25_4),
        .conv25_5(conv25_5), .conv25_6(conv25_6), .conv25_7(conv25_7), .conv25_8(conv25_8),
        .conv25_9(conv25_9), .conv25_10(conv25_10),
        .final_output(final_output)
    );

    task automatic check_outputs;
        begin
            exp_conv[0] = in_1  + in_2  + in_3;
            exp_conv[1] = in_4  + in_5  + in_6;
            exp_conv[2] = in_7  + in_8  + in_9;
            exp_conv[3] = in_10 + in_11 + in_12;
            exp_conv[4] = in_13 + in_14 + in_15;
            exp_conv[5] = in_16 + in_17 + in_18;
            exp_conv[6] = in_19 + in_20 + in_21;
            exp_conv[7] = in_22 + in_23 + in_24;
            exp_conv[8] = in_25 + in_26 + in_27;
            exp_conv[9] = in_28 + in_29 + in_30;

            exp_total = in_1 + in_2 + in_3 + in_4 + in_5 + in_6 + in_7 + in_8 +
                        in_9 + in_10 + in_11 + in_12 + in_13 + in_14 + in_15 + in_16 +
                        in_17 + in_18 + in_19 + in_20 + in_21 + in_22 + in_23 + in_24 +
                        in_25 + in_26 + in_27 + in_28 + in_29 + in_30 + in_31 + in_32;

            #1;

            if (conv25_1  !== exp_conv[0][17:0]) begin errors = errors + 1; $display("Mismatch conv25_1");  end
            if (conv25_2  !== exp_conv[1][17:0]) begin errors = errors + 1; $display("Mismatch conv25_2");  end
            if (conv25_3  !== exp_conv[2][17:0]) begin errors = errors + 1; $display("Mismatch conv25_3");  end
            if (conv25_4  !== exp_conv[3][17:0]) begin errors = errors + 1; $display("Mismatch conv25_4");  end
            if (conv25_5  !== exp_conv[4][17:0]) begin errors = errors + 1; $display("Mismatch conv25_5");  end
            if (conv25_6  !== exp_conv[5][17:0]) begin errors = errors + 1; $display("Mismatch conv25_6");  end
            if (conv25_7  !== exp_conv[6][17:0]) begin errors = errors + 1; $display("Mismatch conv25_7");  end
            if (conv25_8  !== exp_conv[7][17:0]) begin errors = errors + 1; $display("Mismatch conv25_8");  end
            if (conv25_9  !== exp_conv[8][17:0]) begin errors = errors + 1; $display("Mismatch conv25_9");  end
            if (conv25_10 !== exp_conv[9][17:0]) begin errors = errors + 1; $display("Mismatch conv25_10"); end

            if (final_output !== exp_total[17:0]) begin
                errors = errors + 1;
                $display("Mismatch final_output: got=%0d expected=%0d", final_output, exp_total[17:0]);
            end
        end
    endtask

    task automatic set_all_inputs(input signed [17:0] value);
        begin
            in_1=value; in_2=value; in_3=value; in_4=value; in_5=value; in_6=value; in_7=value; in_8=value;
            in_9=value; in_10=value; in_11=value; in_12=value; in_13=value; in_14=value; in_15=value; in_16=value;
            in_17=value; in_18=value; in_19=value; in_20=value; in_21=value; in_22=value; in_23=value; in_24=value;
            in_25=value; in_26=value; in_27=value; in_28=value; in_29=value; in_30=value; in_31=value; in_32=value;
        end
    endtask

    task automatic set_onehot_input(
        input integer idx,
        input signed [17:0] value
    );
        begin
            set_all_inputs(18'sd0);
            case (idx)
                1:  in_1  = value;  2:  in_2  = value;  3:  in_3  = value;  4:  in_4  = value;
                5:  in_5  = value;  6:  in_6  = value;  7:  in_7  = value;  8:  in_8  = value;
                9:  in_9  = value;  10: in_10 = value;  11: in_11 = value; 12: in_12 = value;
                13: in_13 = value; 14: in_14 = value; 15: in_15 = value; 16: in_16 = value;
                17: in_17 = value; 18: in_18 = value; 19: in_19 = value; 20: in_20 = value;
                21: in_21 = value; 22: in_22 = value; 23: in_23 = value; 24: in_24 = value;
                25: in_25 = value; 26: in_26 = value; 27: in_27 = value; 28: in_28 = value;
                29: in_29 = value; 30: in_30 = value; 31: in_31 = value; 32: in_32 = value;
                default: begin end
            endcase
        end
    endtask

    initial begin
        errors = 0;
        seed = 32'h4A3C_2B1D;

        // Case 1: all zeros
        set_all_inputs(18'sd0);
        check_outputs();

        // Case 2: all ones
        set_all_inputs(18'sd1);
        check_outputs();

        // Case 3: all minus ones
        set_all_inputs(-18'sd1);
        check_outputs();

        // Case 4: max positive on all inputs
        set_all_inputs(MAX18);
        check_outputs();

        // Case 5: max negative on all inputs
        set_all_inputs(MIN18);
        check_outputs();

        // Case 6: alternating pattern
        in_1=18'sd100;   in_2=-18'sd100;  in_3=18'sd300;   in_4=-18'sd300;
        in_5=18'sd500;   in_6=-18'sd500;  in_7=18'sd700;   in_8=-18'sd700;
        in_9=18'sd900;   in_10=-18'sd900; in_11=18'sd1100; in_12=-18'sd1100;
        in_13=18'sd1300; in_14=-18'sd1300;in_15=18'sd1500; in_16=-18'sd1500;
        in_17=18'sd1700; in_18=-18'sd1700;in_19=18'sd1900; in_20=-18'sd1900;
        in_21=18'sd2100; in_22=-18'sd2100;in_23=18'sd2300; in_24=-18'sd2300;
        in_25=18'sd2500; in_26=-18'sd2500;in_27=18'sd2700; in_28=-18'sd2700;
        in_29=18'sd2900; in_30=-18'sd2900;in_31=18'sd3100; in_32=-18'sd3100;
        check_outputs();

        // Case 7: verify in_31/in_32 integration path explicitly
        set_all_inputs(18'sd0);
        in_31 = MAX18;
        in_32 = MIN18;
        check_outputs();

        // Case 8: one-hot sweep for each input with +/- impulse
        for (onehot_idx = 1; onehot_idx <= 32; onehot_idx = onehot_idx + 1) begin
            impulse_val = (onehot_idx % 2) ? 18'sd12345 : -18'sd12345;
            set_onehot_input(onehot_idx, impulse_val);
            check_outputs();
        end

        // Random regression (broad coverage, deterministic seed)
        for (i = 0; i < 5000; i = i + 1) begin
            in_1  = $random(seed); in_2  = $random(seed); in_3  = $random(seed); in_4  = $random(seed);
            in_5  = $random(seed); in_6  = $random(seed); in_7  = $random(seed); in_8  = $random(seed);
            in_9  = $random(seed); in_10 = $random(seed); in_11 = $random(seed); in_12 = $random(seed);
            in_13 = $random(seed); in_14 = $random(seed); in_15 = $random(seed); in_16 = $random(seed);
            in_17 = $random(seed); in_18 = $random(seed); in_19 = $random(seed); in_20 = $random(seed);
            in_21 = $random(seed); in_22 = $random(seed); in_23 = $random(seed); in_24 = $random(seed);
            in_25 = $random(seed); in_26 = $random(seed); in_27 = $random(seed); in_28 = $random(seed);
            in_29 = $random(seed); in_30 = $random(seed); in_31 = $random(seed); in_32 = $random(seed);
            check_outputs();
        end

        if (errors == 0) begin
            $display("TEST PASSED: no mismatches.");
        end else begin
            $display("TEST FAILED: %0d mismatches.", errors);
        end

        $finish;
    end

endmodule
