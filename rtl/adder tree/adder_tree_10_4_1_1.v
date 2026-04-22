`timescale 1ns / 1ps

module adder_tree_10_4_1_1(
    input wire signed [17:0] in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8,
    input wire signed [17:0] in_9, in_10, in_11, in_12, in_13, in_14, in_15, in_16,
    input wire signed [17:0] in_17, in_18, in_19, in_20, in_21, in_22, in_23, in_24,
    input wire signed [17:0] in_25, in_26, in_27, in_28, in_29, in_30, in_31, in_32,

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


// Assign outputs directly from the high-precision L1 nets
    assign conv25_1  = L1_1[20:3];
    assign conv25_2  = L1_2[20:3];
    assign conv25_3  = L1_3[20:3];
    assign conv25_4  = L1_4[20:3];
    assign conv25_5  = L1_5[20:3];
    assign conv25_6  = L1_6[20:3];
    assign conv25_7  = L1_7[20:3]; 
    assign conv25_8  = L1_8[20:3];
    assign conv25_9  = L1_9[20:3];
    assign conv25_10 = L1_10[20:3];

    // Internal accumulation wires (typically for DSP high-precision paths)
    wire signed [47:0] L1_1, L1_2, L1_3, L1_4, L1_5;
    wire signed [47:0] L1_6, L1_7, L1_8, L1_9, L1_10;

    // --- Layer 1: 32 inputs -> 10 groups ---
    adder_3 layer_1__1  (.add_1(in_1),  .add_2(in_2),  .add_3(in_3),  .final_adder_out(), .internal_adder_out(L1_1)  );
    adder_3 layer_1__2  (.add_1(in_4),  .add_2(in_5),  .add_3(in_6),  .final_adder_out(), .internal_adder_out(L1_2)  );
    adder_3 layer_1__3  (.add_1(in_7),  .add_2(in_8),  .add_3(in_9),  .final_adder_out(), .internal_adder_out(L1_3)  );
    adder_3 layer_1__4  (.add_1(in_10), .add_2(in_11), .add_3(in_12), .final_adder_out(), .internal_adder_out(L1_4)  );
    adder_3 layer_1__5  (.add_1(in_13), .add_2(in_14), .add_3(in_15), .final_adder_out(), .internal_adder_out(L1_5)  );
    adder_3 layer_1__6  (.add_1(in_16), .add_2(in_17), .add_3(in_18), .final_adder_out(), .internal_adder_out(L1_6)  );
    adder_3 layer_1__7  (.add_1(in_19), .add_2(in_20), .add_3(in_21), .final_adder_out(), .internal_adder_out(L1_7)  );
    adder_3 layer_1__8  (.add_1(in_22), .add_2(in_23), .add_3(in_24), .final_adder_out(), .internal_adder_out(L1_8)  );
    adder_3 layer_1__9  (.add_1(in_25), .add_2(in_26), .add_3(in_27), .final_adder_out(), .internal_adder_out(L1_9)  );
    adder_3 layer_1__10 (.add_1(in_28), .add_2(in_29), .add_3(in_30), .final_adder_out(), .internal_adder_out(L1_10) );

    // --- Layer 2: 10 outputs -> 4 intermediate ---
    wire signed [47:0] L2_1, L2_2, L2_3, L2_4;

    adder_3 layer_2__1 (.add_1(L1_1[20:3]),  .add_2(L1_2[20:3]),  .add_3(L1_3[20:3]), .final_adder_out(), .internal_adder_out(L2_1) );
    adder_3 layer_2__2 (.add_1(L1_4[20:3]),  .add_2(L1_5[20:3]),  .add_3(L1_6[20:3]), .final_adder_out(), .internal_adder_out(L2_2) );
    adder_3 layer_2__3 (.add_1(L1_7[20:3]),  .add_2(L1_8[20:3]),  .add_3(L1_9[20:3]), .final_adder_out(), .internal_adder_out(L2_3) );
    adder_3 layer_2__4 (.add_1(L1_10[20:3]), .add_2(in_31),       .add_3(in_32),      .final_adder_out(), .internal_adder_out(L2_4) );

    // --- Layers 3 & 4 ---
    wire signed [47:0] L3, L4;

    adder_3 layer_3 (
        .add_1(L2_1[20:3]), 
        .add_2(L2_2[20:3]), 
        .add_3(L2_3[20:3]), 
        .final_adder_out(),  
        .internal_adder_out(L3)  
    );

    adder_3 layer_4 (
        .add_1(L3[20:3]),      // Sliced L3 correctly
        .add_2(L2_4[20:3]),    // Replaced L4 with the missing L2_4 input
        .add_3(18'sd0),        // Used signed 0 
        .final_adder_out(),    // Disconnected final_adder_out
        .internal_adder_out(L4) // Driven internal_adder_out correctly to L4
    );

    assign final_output = L4[20:3];

endmodule
