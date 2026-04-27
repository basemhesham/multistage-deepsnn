module Batch_Norm #(parameter DATA_WIDTH = 18 /*,parameter signed A = 18'd4 ,parameter signed B = 18'd2 */)(

input wire signed [DATA_WIDTH-1:0] Batch_Norm_in ,
input wire signed [DATA_WIDTH-1:0] mult_wight ,
input wire signed [DATA_WIDTH-1:0] add_wight ,
output wire signed [DATA_WIDTH-1:0] Batch_Norm_out 
);

//assign Batch_Norm_out = (Batch_Norm_in*A) + B ;
wire signed [36:0] mult_Add_result ; // 37 bit 

assign Batch_Norm_out = mult_Add_result[36:19];

xbip_dsp48_macro_0 dsp48_inst_0 (
  .A(Batch_Norm_in),  // input wire [17 : 0] A
  .B(mult_wight),  // input wire [17 : 0] B
  .C(add_wight),  // input wire [17 : 0] C
  .P(mult_Add_result)  // output wire [36 : 0] P
);

endmodule
