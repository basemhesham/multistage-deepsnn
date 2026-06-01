module LIF #(parameter DATA_WIDTH = 18 , parameter signed threshold = 18'd5 , parameter signed zero = 18'd0) (
input wire clk,
input wire rst,
input wire signed [DATA_WIDTH-1:0] in_pool ,
output reg spike 
);

reg signed [DATA_WIDTH-1:0] mem_reg ;
reg  spike_reg ;

wire signed [DATA_WIDTH-1:0] mux_out , mem_leak , new_mem , mem_input_add_trnuc ; // 18 bits
wire signed [DATA_WIDTH:0] mem_input_add ; // 19 bits
reg spike_int ; // internal spike ;

always @(posedge clk)
begin
 if(rst)
  begin
   mem_reg <= 'd0;
   spike_reg<= 1'b0;
  end
 else
  begin
  mem_reg <= new_mem ;
  spike_reg <= spike_int ;
  end

end



assign mux_out = spike_reg? threshold : zero;
assign mem_leak = mem_reg >>> 1 ; // step 1 : decay  (assume beta = 0.5)  means divided by 2 
assign mem_input_add = mem_leak + in_pool;
assign mem_input_add_trnuc = mem_input_add[DATA_WIDTH:1];
assign new_mem = mem_input_add_trnuc - mux_out ;

always @(*)
 begin
  if(new_mem >= threshold)
   begin
    spike = 1'b1 ;
	spike_int = 1'b1;
   end
  else
   begin
    spike = 1'b0 ;
	spike_int = 1'b0;
   end
  
 end

endmodule
