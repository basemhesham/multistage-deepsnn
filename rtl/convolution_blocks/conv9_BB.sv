<<<<<<<< HEAD:rtl/conv9_BB25.sv
module cov9_BB (
    input wire logic  [17:0] P [0:7],
    input wire logic  [17:0] Q [0:7],
    output logic [39:0] Out
);

    // Stage 1: Multiplications (36-bit)
    wire [35:0] m [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : MULT_STAGE
            assign m[i] = P[i] * Q[i];
        end
    endgenerate

    // Stage 2: First level additions
    wire [36:0] s1_0 = m[0] + m[1];
    wire [36:0] s1_1 = m[2] + m[3];
    wire [36:0] s1_2 = m[4] + m[5];
    wire [36:0] s1_3 = m[6] + m[7];

    // m[8] goes forward directly

    // Stage 3: Second level
    wire [37:0] s2_0 = s1_0 + s1_1;
    wire [37:0] s2_1 = s1_2 + s1_3;

    // Stage 4: Third level
    wire [38:0] s3 = s2_0 + s2_1;

    // Final stage
    assign Out = s3;  // 40-bit result

endmodule
========
module cov9_BB (
    input wire logic  [17:0] P [0:7],
    input wire logic  [17:0] Q [0:7],
    output logic [39:0] Out
);

    // Stage 1: Multiplications (36-bit)
    wire [35:0] m [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : MULT_STAGE
            assign m[i] = P[i] * Q[i];
        end
    endgenerate

    // Stage 2: First level additions
    wire [36:0] s1_0 = m[0] + m[1];
    wire [36:0] s1_1 = m[2] + m[3];
    wire [36:0] s1_2 = m[4] + m[5];
    wire [36:0] s1_3 = m[6] + m[7];

    // m[8] goes forward directly

    // Stage 3: Second level
    wire [37:0] s2_0 = s1_0 + s1_1;
    wire [37:0] s2_1 = s1_2 + s1_3;

    // Stage 4: Third level
    wire [38:0] s3 = s2_0 + s2_1;

    // Final stage
    assign Out = s3;  // 40-bit result

endmodule
>>>>>>>> d36020b (update convolution blocks):rtl/convolution_blocks/conv9_BB.sv
