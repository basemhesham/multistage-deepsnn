module tb;

    logic            stage_sel;
    logic   [0:31]   shaaban_out [0:31];
    logic   [0:31]   mem_mapped [0:3199];

mem_maping_1_2 DUT (
stage_sel,
shaaban_out,
mem_mapped 
);


initial begin

    stage_sel = 1;
    for (int m=0; m<3; m++) begin
        shaaban_out[m] = m;
    end


#100;


    stage_sel = 0;
    for (int m=0; m<32; m++) begin
        shaaban_out[m] = m;
    end




#100;



$stop();
end

endmodule