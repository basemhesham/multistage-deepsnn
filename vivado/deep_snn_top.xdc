## deep_snn_top timing constraints
## Target clock: 40 MHz -> 25.000 ns period

create_clock -name clk_40mhz -period 25.000 [get_ports clk]
set_clock_uncertainty -setup 0.250 [get_clocks clk_40mhz]
set_clock_uncertainty -hold 0.000 [get_clocks clk_40mhz]

## Board-level I/O timing placeholders for synchronous top-level ports.
## Replace these values when the real external interface timing is known.
set_input_delay -clock [get_clocks clk_40mhz] -max 2.000 [get_ports {
    rst
    arst_n
    enable
    pixel_mem_wr_en
    pixel_mem_wr_addr[*]
    pixel_mem_wr_data[*]
}]
set_input_delay -clock [get_clocks clk_40mhz] -min 0.000 [get_ports {
    rst
    arst_n
    enable
    pixel_mem_wr_en
    pixel_mem_wr_addr[*]
    pixel_mem_wr_data[*]
}]

set_output_delay -clock [get_clocks clk_40mhz] -max 2.000 [get_ports {
    spike_out[*]
    class_logits[*]
    classifier_done
    classifier_busy
    snn_done
    done
}]
set_output_delay -clock [get_clocks clk_40mhz] -min 0.000 [get_ports {
    spike_out[*]
    class_logits[*]
    classifier_done
    classifier_busy
    snn_done
    done
}]

## arst_n is an external asynchronous reset input.
set_false_path -from [get_ports arst_n]
