`timescale 1us/100ns

// ============================================================
// Testbench: serial_adder_tb
// Description: Verifies the functionality of a serial adder
//              by driving random inputs and comparing outputs
//              against a golden reference model.
// ============================================================
module serial_adder_tb();

    // --------------------------------------------------------
    // Parameters
    // --------------------------------------------------------
    parameter BITS = 8;  // Bit width of the adder inputs

    // --------------------------------------------------------
    // DUT Interface Signals
    // --------------------------------------------------------
    logic clk;
    logic reset_n;
    logic serial_a;
    logic serial_b;
    logic valid_in;
    logic busy;
    logic valid_out;
    logic serial_sum;

    // --------------------------------------------------------
    // Scorecard Counters
    // --------------------------------------------------------
    int pass_count;
    int fail_count;

    // --------------------------------------------------------
    // Clock Generation: 1 MHz (period = 1us)
    // --------------------------------------------------------
    initial begin
        clk = 1'b0;
        forever #0.5 clk = ~clk;
    end

    // --------------------------------------------------------
    // DUT Instantiation
    // --------------------------------------------------------
    simple_adder #(
        .N(BITS)
    ) DUT (
        .clk    (clk),
        .rst_n  (reset_n),
        .in_a   (serial_a),
        .in_b   (serial_b),
        .in_vld (valid_in),
        .o_busy (busy),
        .o_vld  (valid_out),
        .o_sum  (serial_sum)
    );

    // --------------------------------------------------------
    // Mailboxes for inter-task communication
    // --------------------------------------------------------
    mailbox #() gen_to_drv_mb;      // Generator -> Driver
    mailbox #() drv_to_gm_mb;       // Driver    -> Golden Model
    mailbox #() gm_result_mb;       // Golden Model results
    mailbox #() monitor_result_mb;  // Monitor captures

    // --------------------------------------------------------
    // Waveform Dump + Simulation Timeout Watchdog
    // --------------------------------------------------------
    initial begin
        $dumpfile("waveform.vcd");
        $vcdpluson;
        $dumpvars(0, serial_adder_tb);

        // Allow up to 300 clock cycles before forcing finish
        repeat(300) @(negedge clk);
        $display("[WATCHDOG] Simulation timed out!");
        $finish();
    end

    // --------------------------------------------------------
    // Main Test Sequence
    // --------------------------------------------------------
    initial begin
        // Initialise mailboxes
        gen_to_drv_mb     = new();
        drv_to_gm_mb      = new();
        gm_result_mb      = new();
        monitor_result_mb = new();

        // Initialise all inputs to safe defaults
        reset_n  = 1'b0;
        serial_a = 1'b0;
        serial_b = 1'b0;
        valid_in = 1'b0;

        // Hold reset for two falling edges then release
        repeat(2) @(negedge clk);
        reset_n = 1'b1;
        @(negedge clk);

        // Run 10 independent test transactions
        for (int test_num = 0; test_num < 10; test_num++) begin
            // Wait for DUT to be free before starting a new transaction
            wait(!busy);

            // Launch all verification tasks concurrently for this transaction
            fork
                gen_random_inputs();
                drive_inputs(serial_a, serial_b, valid_in);
                capture_inputs(serial_a, serial_b, valid_in);
                compute_expected();
                capture_output(valid_out, serial_sum);
                compare_results(pass_count, fail_count);
            join
        end

        // ---- Final Simulation Report ----
        $display("=========================================");
        $display("  SIMULATION COMPLETE");
        $display("  PASS : %0d", pass_count);
        $display("  FAIL : %0d", fail_count);
        $display("=========================================");
        $stop;
    end


    // ============================================================
    // TASK: gen_random_inputs
    // Generates two random BITS-wide operands and pushes them
    // into the generator mailbox for the driver to consume.
    // ============================================================
    task gen_random_inputs();
        logic [BITS-1:0] operand_a;
        logic [BITS-1:0] operand_b;

        operand_a = $urandom();
        operand_b = $urandom();

        gen_to_drv_mb.put(operand_a);
        gen_to_drv_mb.put(operand_b);
    endtask


    // ============================================================
    // TASK: drive_inputs
    // Retrieves operands from the mailbox and serially shifts
    // them into the DUT MSB-first, following the valid protocol.
    // ============================================================
    task drive_inputs(output logic serial_a,
                      output logic serial_b,
                      output logic valid_in);

        logic [BITS-1:0] operand_a;
        logic [BITS-1:0] operand_b;
        bit   started;

        // Collect operands produced by the generator
        gen_to_drv_mb.get(operand_a);
        gen_to_drv_mb.get(operand_b);

        for (int cycle = 0; cycle < BITS + 2; cycle++) begin
            if (!started) begin
                // First cycle: assert valid, send zeros as preamble
                valid_in = 1'b1;
                serial_a = 1'b0;
                serial_b = 1'b0;
                started  = 1'b1;

            end else if (cycle <= BITS) begin
                // Data cycles: send bits MSB-first
                serial_a = operand_a[BITS - cycle];
                serial_b = operand_b[BITS - cycle];

                // Forward driven bits to the golden model
                drv_to_gm_mb.put(operand_a[BITS - cycle]);
                drv_to_gm_mb.put(operand_b[BITS - cycle]);

            end else begin
                // De-assert valid after all data bits sent
                valid_in = 1'b0;
            end

            @(negedge clk);
        end

        // Clean up for next transaction
        started  = 1'b0;
        valid_in = 1'b0;
    endtask


    // ============================================================
    // TASK: capture_inputs
    // Monitors the DUT's serial input pins and displays the
    // reconstructed parallel values for debug visibility.
    // ============================================================
    task capture_inputs(input logic serial_a,
                        input logic serial_b,
                        input logic valid_in);

        bit [BITS-1:0] reconstructed_a;
        bit [BITS-1:0] reconstructed_b;

        // Wait until the driver asserts valid
        wait(valid_in);
        @(negedge clk);

        // Sample one bit per clock for BITS cycles
        for (int i = 0; i < BITS; i++) begin
            reconstructed_a[BITS-1-i] = serial_a;
            reconstructed_b[BITS-1-i] = serial_b;
            @(negedge clk);
        end

        $display("[MONITOR IN]  A = %0d  |  B = %0d",
                 reconstructed_a, reconstructed_b);
    endtask


    // ============================================================
    // TASK: compute_expected
    // Rebuilds the two operands from the driver mailbox bits,
    // computes the reference sum, and stores it for the checker.
    // ============================================================
    task compute_expected();
        bit [BITS-1:0] operand_a;
        bit [BITS-1:0] operand_b;
        bit [BITS:0]   expected_sum;
        logic          bit_a;
        logic          bit_b;

        // Reassemble operands from individually driven bits
        for (int i = 0; i < BITS; i++) begin
            drv_to_gm_mb.get(bit_a);
            drv_to_gm_mb.get(bit_b);
            operand_a = {operand_a[BITS-2:0], bit_a};
            operand_b = {operand_b[BITS-2:0], bit_b};
        end

        expected_sum = operand_a + operand_b;
        gm_result_mb.put(expected_sum);
    endtask


    // ============================================================
    // TASK: capture_output
    // Waits for the DUT to assert o_vld then captures the
    // (BITS+1)-wide serial result, MSB-first.
    // ============================================================
    task capture_output(input logic valid_out,
                        input logic serial_sum);

        bit [BITS:0] captured_sum;

        // Block until DUT signals a valid result
        wait(valid_out);

        for (int i = 0; i < BITS + 1; i++) begin
            captured_sum[BITS - i] = serial_sum;
            @(negedge clk);
        end

        monitor_result_mb.put(captured_sum);
        $display("[MONITOR OUT] Sum = %0d", captured_sum);
    endtask


    // ============================================================
    // TASK: compare_results
    // Pulls values from both result mailboxes and checks whether
    // the DUT output matches the golden model expectation.
    // ============================================================
    task compare_results(ref int pass_count, ref int fail_count);
        bit [BITS:0] expected;
        bit [BITS:0] actual;

        gm_result_mb.get(expected);
        monitor_result_mb.get(actual);

        if (actual == expected) begin
            $display("[CHECKER]     PASS  (expected=%0d, got=%0d)",
                     expected, actual);
            pass_count++;
        end else begin
            $display("[CHECKER]     FAIL  (expected=%0d, got=%0d)",
                     expected, actual);
            fail_count++;
        end
    endtask

endmodule