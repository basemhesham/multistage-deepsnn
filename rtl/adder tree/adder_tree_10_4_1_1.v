`timescale 1ns / 1ps

module adder_tree_10_4_1_1(
    input  wire signed [17:0] in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8,
    input  wire signed [17:0] in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16,
    input  wire signed [17:0] in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24,
    input  wire signed [17:0] in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32,

    output wire signed [17:0] conv25_1,
    output wire signed [17:0] conv25_2,
    output wire signed [17:0] conv25_3,
    output wire signed [17:0] conv25_4,
    output wire signed [17:0] conv25_5,
    output wire signed [17:0] conv25_6,
    output wire signed [17:0] conv25_7,
    output wire signed [17:0] conv25_8,
    output wire signed [17:0] conv25_9,
    output wire signed [17:0] conv25_10,

    output wire signed [17:0] final_output
    );


    wire signed [19:0] L1_1; 
    wire signed [19:0] L1_2; 
    wire signed [19:0] L1_3; 
    wire signed [19:0] L1_4; 
    wire signed [19:0] L1_5; 
    wire signed [19:0] L1_6; 
    wire signed [19:0] L1_7; 
    wire signed [19:0] L1_8; 
    wire signed [19:0] L1_9; 
    wire signed [19:0] L1_10; 


    assign conv25_1  = L1_1[19:1];
    assign conv25_2  = L1_2[19:1];
    assign conv25_3  = L1_3[19:1];
    assign conv25_4  = L1_4[19:1];
    assign conv25_5  = L1_5[19:1];
    assign conv25_6  = L1_6[19:1];
    assign conv25_7  = L1_7[19:1]; 
    assign conv25_8  = L1_8[19:1];
    assign conv25_9  = L1_9[19:1];
    assign conv25_10 = L1_10[19:1];


    // Layer_1:   18 -> 21
    // Layer_2:   21 -> 24




    // --- Layer 1: 32 inputs -> 10 groups ---
    // (Each adder_3 takes 3 inputs. 10 adders = 30 inputs. 
    // Inputs 31 and 32 will be handled in the separate logic.
    
    adder_layer1     layer_1__1  (.add_1(in_1),  .add_2(in_2),  .add_3(in_3),  .adder_out(L1_1)  );
    adder_layer1     layer_1__2  (.add_1(in_4),  .add_2(in_5),  .add_3(in_6),  .adder_out(L1_2)  );
    adder_layer1     layer_1__3  (.add_1(in_7),  .add_2(in_8),  .add_3(in_9),  .adder_out(L1_3)  );
    adder_layer1     layer_1__4  (.add_1(in_10), .add_2(in_11), .add_3(in_12), .adder_out(L1_4)  );
    adder_layer1     layer_1__5  (.add_1(in_13), .add_2(in_14), .add_3(in_15), .adder_out(L1_5)  );
    adder_layer1     layer_1__6  (.add_1(in_16), .add_2(in_17), .add_3(in_18), .adder_out(L1_6)  );
    adder_layer1     layer_1__7  (.add_1(in_19), .add_2(in_20), .add_3(in_21), .adder_out(L1_7)  );
    adder_layer1     layer_1__8  (.add_1(in_22), .add_2(in_23), .add_3(in_24), .adder_out(L1_8)  );
    adder_layer1     layer_1__9  (.add_1(in_25), .add_2(in_26), .add_3(in_27), .adder_out(L1_9)  );
    adder_layer1     layer_1__10 (.add_1(in_28), .add_2(in_29), .add_3(in_30), .adder_out(L1_10) );




    // --- Layer 2: 10 outputs -> 4 intermediate ---
    wire signed [21:0] L2_1, L2_2, L2_3;
    wire signed [20:0] L2_4;


    adder_layer2     layer_2__1 (.add_1(L1_1),  .add_2(L1_2),  .add_3(L1_3),  .adder_out(L2_1)  );
    adder_layer2     layer_2__2 (.add_1(L1_4),  .add_2(L1_5),  .add_3(L1_6),  .adder_out(L2_2)  );
    adder_layer2     layer_2__3 (.add_1(L1_7),  .add_2(L1_8),  .add_3(L1_9),  .adder_out(L2_3)  );
    //adder_layer2     layer_2__4 (.add_1(L1_10), .add_2(in_31), .add_3(in_32), .adder_out(L2_4) );

xbip_dsp48_macro_0 layer_2__4 (
  .A(in_31),  // input wire [17 : 0] A
  .C(L1_10),  // input wire [19 : 0] C
  .D(in_32),  // input wire [17 : 0] D
  .P(L2_4)  // output wire [20 : 0] P
);

    wire signed [23:0] L3;
    wire signed [24:0] L4;

    dsp48_layer_3       layer_3 (
        .A(L2_1),   // input  wire [21 : 0] A
        .C(L2_2),   // input  wire [21 : 0] C
        .D(L2_3),   // input  wire [21 : 0] D
        .P(L3)      // output wire [23 : 0] P
    );


    dsp48_layer_4       layer_4 (
        .A(L2_4),    // input wire [20 : 0] A
        .D(L3),  // input wire [23 : 0] D
        .P(L4)     // output wire [24 : 0] P
    );



    assign final_output = L4[24:24-18];

endmodule
