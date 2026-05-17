# Create a Vivado project for the current deep_snn_top integration.
#
# Usage from the repository root:
#   C:/Xilinx/Vivado/2018.2/bin/vivado.bat -mode batch -source vivado/create_deep_snn_top_project.tcl
#
# Optional:
#   vivado.bat -mode batch -source vivado/create_deep_snn_top_project.tcl -tclargs -no-synth

set script_path [string map {\\ /} [info script]]
set script_dir  [file dirname $script_path]
set repo_root   [file dirname $script_dir]
set project_dir [file join $repo_root "build" "vivado" "deep_snn_top"]
set reports_dir [file join $repo_root "build" "vivado" "reports"]
set staged_dir  [file join $repo_root "build" "vivado" "src"]
set part_name   "xcvu11p-flga2577-3-e"
set top_name    "deep_snn_top"
set run_synth   1

foreach arg $argv {
    if {$arg eq "-no-synth"} {
        set run_synth 0
    }
}

puts "Script path: $script_path"
puts "Script dir:  $script_dir"
puts "Repo root:   $repo_root"
puts "Project dir: $project_dir"

file mkdir $project_dir
file mkdir $reports_dir
file mkdir $staged_dir

create_project $top_name $project_dir -part $part_name -force
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]

set final_param_dir [file join $repo_root "rtl" "SNN Parameters" "Final Parameters (RTL)"]

set source_files [list \
    [file join $final_param_dir "UNIQUE_CONV1_WEIGHTS.sv"] \
    [file join $final_param_dir "UNIQUE_CONV2_WEIGHTS.sv"] \
    [file join $final_param_dir "UNIQUE_CONV3_WEIGHTS.sv"] \
    [file join $final_param_dir "CONV1_W_MAP_OPT.sv"] \
    [file join $final_param_dir "CONV2_W_MAP_OPT.sv"] \
    [file join $final_param_dir "CONV3_W_MAP_OPT.sv"] \
    [file join $repo_root "rtl" "convolution_blocks" "convDspAddMult.v"] \
    [file join $repo_root "rtl" "convolution_blocks" "conv9.sv"] \
    [file join $repo_root "rtl" "adder tree" "adder_layer1.v"] \
    [file join $repo_root "rtl" "adder tree" "adder_layer2.v"] \
    [file join $repo_root "rtl" "adder tree" "adder_layer3.v"] \
    [file join $repo_root "rtl" "adder tree" "adder_layer4.v"] \
    [file join $repo_root "rtl" "adder tree" "adder_tree_10_4_1_1.v"] \
    [file join $repo_root "rtl" "Shabaan_Adder_connect" "adder_tree_shaaban_connect.sv"] \
    [file join $repo_root "rtl" "Shabaan Unit" "conv_bias_Relu.v"] \
    [file join $repo_root "rtl" "Shabaan Unit" "Batch_Norm.v"] \
    [file join $repo_root "rtl" "Shabaan Unit" "Max_pooling.v"] \
    [file join $repo_root "rtl" "Shabaan Unit" "LIF.v"] \
    [file join $repo_root "rtl" "Shabaan Unit" "shaban_unit_top.v"] \
    [file join $repo_root "rtl" "Controller" "mem_maping_1_2.sv"] \
    [file join $repo_root "rtl" "frame" "frame_mapping_iterations_filters.sv"] \
    [file join $repo_root "rtl" "frame" "frame_input_mapping_brackets.sv"] \
    [file join $repo_root "rtl" "Top" "top.sv"] \
]

foreach src $source_files {
    if {![file exists $src]} {
        error "Required source file does not exist: $src"
    }
}

set staged_files [list]
foreach src $source_files {
    set dst [file join $staged_dir [file tail $src]]
    file copy -force $src $dst
    lappend staged_files $dst
}

add_files -norecurse -fileset sources_1 $staged_files
set_property file_type SystemVerilog [get_files -of_objects [get_filesets sources_1]]
set_property include_dirs [list $staged_dir] [get_filesets sources_1]
set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

puts "Created Vivado project:"
puts "  Project: $project_dir"
puts "  Part:    $part_name"
puts "  Top:     $top_name"
puts "  Sources: [llength $staged_files]"

if {$run_synth} {
    launch_runs synth_1 -jobs 4
    wait_on_run synth_1
    set synth_status [get_property STATUS [get_runs synth_1]]
    puts "synth_1 status: $synth_status"

    if {[string first "Complete" $synth_status] < 0} {
        error "synth_1 did not complete successfully: $synth_status"
    }

    open_run synth_1 -name synth_1
    report_utilization -file [file join $reports_dir "deep_snn_top_utilization_synth.rpt"]
    report_timing_summary -file [file join $reports_dir "deep_snn_top_timing_synth.rpt"]
    report_dsp_utilization -file [file join $reports_dir "deep_snn_top_dsp_synth.rpt"]
}
