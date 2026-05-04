`timescale 1us/100ns
module simple_adder_tb ();

    parameter N = 4;	
    logic rst_n;
    logic in_a;	
    logic in_b;	
    logic in_vld;
    logic o_busy;
    logic o_vld;
    logic o_sum;
    
    logic clk;
    int right_counter;
    int wrong_counter;
    initial begin
        clk = 0;
        forever begin
            #0.5 clk = ~clk;
        end
    end

    //------------------ instantiation
    simple_adder #(
        .N(N)
    ) DUT
    (
        .clk(clk),
        .rst_n(rst_n),
        .in_a(in_a),
        .in_b(in_b),
        .in_vld(in_vld),
        .o_busy(o_busy),
        .o_vld(o_vld),
        .o_sum(o_sum)
    );

    //---------------- mailboxes
    mailbox generator_mb;
    mailbox driver_mb;
    mailbox monitor_out_mb;
    mailbox golden_model_mb;

    // Dump waveforms and finish simulation after a timeout
	initial begin
		$dumpfile("waves.vcd");
		$vcdpluson;
		$dumpvars(0, simple_adder_tb);
		repeat(200) begin
            @(negedge clk);
        end
		$display("TIMEOUT!");
		$finish(); // timeout
	end

    initial begin
        generator_mb    = new();
        driver_mb       = new();
        monitor_out_mb  = new();
        golden_model_mb = new();
        rst_n = 0;
        in_a = 0;
        in_b = 0;
        in_vld = 0;
        @(negedge clk);
        @(negedge clk);
        rst_n = 1;
        @(negedge clk);

        for (int i = 0; i < 10 ; i++) begin
            wait(!o_busy);
            fork
                generator();
                driver(
                    .in_a(in_a),
                    .in_b(in_b),
                    .in_vld(in_vld)
                );
                monitor_in(
                    .in_a(in_a),
                    .in_b(in_b),
                    .in_vld(in_vld)
                );
                golden_model();
                monitor_out(
                    .o_vld(o_vld),
                    .o_sum(o_sum)
                );
                checker_(
                    .right_counter(right_counter),
                    .wrong_counter(wrong_counter)
                );
            join
        end
        $display("number of correct output is : %0d", right_counter);
        $display("number of false output is : %0d", wrong_counter);
        $stop;
    end

    task  generator();
        logic [N-1 :0] a;
        logic [N-1 :0] b;
        if begin
            a = $urandom();
            b = $urandom();
        end
        generator_mb.put(a);
        generator_mb.put(b);
    endtask 

    task driver(output logic in_a, output logic in_b, output logic in_vld);
        logic [N-1 :0] a;
        logic [N-1 :0] b;
        bit start;
        generator_mb.get(a);
        generator_mb.get(b);
        for (int i = 0; i < N+2 ; i++) begin
            if(!start) begin
                in_vld = 1;
                in_a = 0;
                in_b = 0;
                start = 1;
            end
            else if (i <= N) begin
                in_a = a[N-i];
                in_b = b[N-i];
                driver_mb.put(a[N-i]);
                driver_mb.put(b[N-i]);
            end
            else begin
                in_vld = 0;
            end
            @(negedge clk);
        end
        start = 0;
        in_vld = 0;
    endtask

    task golden_model();
        bit [N-1 :0] a;
        bit [N-1 :0] b;
        bit [N :0] sum;
        logic in_a ;
        logic in_b ;
        for (int i = 0; i < N ; i++) begin
            driver_mb.get(in_a);
            driver_mb.get(in_b);
            a = {a[N-2:0],in_a};
            b = {b[N-2:0],in_b};
        end
        sum = a + b;
        golden_model_mb.put(sum);
    endtask

    task  monitor_in(input logic in_a, input logic in_b, input logic in_vld);
        bit [N-1 :0] a;
        bit [N-1 :0] b;
        wait(in_vld);
        @(negedge clk);
        for (int i = 0; i < N ; i++ ) begin
            a[N-1-i] = in_a;
            b[N-1-i] = in_b;
            @(negedge clk);
        end
        $display("input a is : %0d ,, input b is : %0d",a,b);
    endtask

    task  monitor_out(input logic o_vld, input logic o_sum);
        bit [N :0] sum;
        wait(o_vld);
        for (int i = 0; i < N+1 ; i++ ) begin
            sum[N-i] = o_sum;
            @(negedge clk);
        end
        monitor_out_mb.put(sum);
        $display("Output is : %0d",sum);
    endtask

    task  checker_(ref int right_counter, ref int wrong_counter);
        bit [N :0] gm_sum;
        bit [N :0] model_sum;
        golden_model_mb.get(gm_sum);
        monitor_out_mb.get(model_sum);
        if(gm_sum == model_sum)begin
            right_counter++;
        end
        else begin
            wrong_counter++;
        end
    endtask 
endmodule