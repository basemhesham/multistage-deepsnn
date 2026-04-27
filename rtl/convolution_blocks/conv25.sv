module cov25 (
	input wire logic  [17:0] P [0:24],
    input wire logic  [17:0] Q [0:24],
    output logic [39:0] Pixel_Out
);

  wire [39:0] Out [0:4];
  wire [39:0] m;

  cov9_BB cov9_1( .P(P[0:7]), .Q(Q[0:7]), .Out(Out[0]));
  cov9_BB cov9_2( .P(P[8:15]), .Q(Q[8:15]), .Out(Out[1]));
  cov9_BB cov9_3( .P(P[16:23]), .Q(Q[16:23]), .Out(Out[2]));

  assign m = P[24] * Q[24];

  assign Out[3] = Out[0] + Out[1];
  assign Out[4] = Out[2] + m ;

  assign Pixel_Out = Out[3] + Out[4];


endmodule