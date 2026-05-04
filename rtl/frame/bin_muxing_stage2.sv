module bin_muxing_stage2 (
  input  wire  din  [1024],
  output logic dout [9][4][64]
);

  // Dimensions for the 4x4 -> 3x3 sliding window muxing
  localparam int NUM_BLOCKS    = 64;
  localparam int BLOCK_SIDE    = 4;
  localparam int WINDOW_SIDE   = 3;
  localparam int BLOCK_STRIDE  = BLOCK_SIDE * BLOCK_SIDE; // 16 bits per block

  always_comb begin
    for (int i = 0; i < NUM_BLOCKS; i++) begin
      for (int k = 0; k < 4; k++) begin
        for (int j = 0; j < (WINDOW_SIDE * WINDOW_SIDE); j++) begin
          // Indexing breakdown:
          // i * BLOCK_STRIDE       : Offset for the current 16-bit block
          // (j/3)*BLOCK_SIDE + j%3 : 3x3 local coordinates within the block
          // (k/2)*BLOCK_SIDE + k%2 : Sliding window offset (k: 0=TL, 1=TR, 2=BL, 3=BR)
          
          int base_idx   = i * BLOCK_STRIDE;
          int window_idx = (j / WINDOW_SIDE) * BLOCK_SIDE + (j % WINDOW_SIDE);
          int slide_off  = (k / 2) * BLOCK_SIDE + (k % 2);

          dout[j][k][i] = din[base_idx + window_idx + slide_off];
        end
      end
    end
  end

endmodule