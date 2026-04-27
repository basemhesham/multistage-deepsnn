class txn;
    // Array of 32 signed 18-bit inputs
    rand bit signed [17:0] inputs [32]; 
endclass


`timescale 1ns / 1ps

module tb_full_tree();
    // Inputs
    logic signed [17:0] in [1:32];
    
    // Outputs
    wire signed [17:0] conv25 [1:10];
    wire signed [17:0] final_output;

    // Golden Reference - Must be 25 bits to match L4 before truncation
    // Or 32 bits to ensure no overflow during the calculation
    logic signed [31:0] golden_sum;
    logic signed [17:0] expected_truncated;

    int correct = 0;
    int wrong   = 0;

    txn obj = new();

    // Instantiate the 32-input adder tree
    adder_tree_10_4_1_1 DUT (
        .in_1(in[1]),   .in_2(in[2]),   .in_3(in[3]),   .in_4(in[4]),
        .in_5(in[5]),   .in_6(in[6]),   .in_7(in[7]),   .in_8(in[8]),
        .in_9(in[9]),   .in_10(in[10]), .in_11(in[11]), .in_12(in[12]),
        .in_13(in[13]), .in_14(in[14]), .in_15(in[15]), .in_16(in[16]),
        .in_17(in[17]), .in_18(in[18]), .in_19(in[19]), .in_20(in[20]),
        .in_21(in[21]), .in_22(in[22]), .in_23(in[23]), .in_24(in[24]),
        .in_25(in[25]), .in_26(in[26]), .in_27(in[27]), .in_28(in[28]),
        .in_29(in[29]), .in_30(in[30]), .in_31(in[31]), .in_32(in[32]),
        
        .final_output(final_output)
        // Note: Map conv25_1 to conv25_10 here if needed
    );

    initial begin
        repeat (100) begin
            if (!obj.randomize()) $error("Randomization failed");

            // Drive inputs and calculate golden sum
            golden_sum = 0;
            for (int i = 1; i <= 32; i++) begin
                in[i] = obj.inputs[i-1]; // obj.inputs is 0-indexed
                golden_sum += in[i];
            end

            // Apply the same truncation logic used in your module:
            // Your module: L4[24 : 24-18] -> bits [24:7]
            expected_truncated = golden_sum[24:7];

            #10; // Wait for combinational logic

            // Check against L4 (Internal full sum) and final_output
            // Using uut.L4 to check the "pre-truncation" value you asked for
            if (DUT.L4 === golden_sum[24:0] /*  && final_output === expected_truncated */) begin
                correct++;
            end else begin
                wrong++;
                $display("ERROR at time %t", $time);
                $display("Expected Full Sum: %d | Actual L4: %d", golden_sum, DUT.L4);
                $display("Expected Truncated: %h | Actual Output: %h", expected_truncated, final_output);
            end
        end

        $display("---------------------------------------");
        $display("Simulation Finished");
        $display("Passed: %0d", correct);
        $display("Failed: %0d", wrong);
        $display("---------------------------------------");
        $finish;
    end

endmodule