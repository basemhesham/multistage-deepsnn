# Running Vivado With This TCL Script

This folder contains `create_deep_snn_top_project.tcl`, which creates a Vivado
project for the current `deep_snn_top` integration.

## What The Script Does

- Creates a Vivado project at:
  `build/vivado/deep_snn_top/deep_snn_top.xpr`
- Targets:
  `xcvu11p-flga2577-3-e`
- Sets the top module to:
  `deep_snn_top`
- Adds the RTL files needed by the current top datapath.
- Copies those RTL files into:
  `build/vivado/src`
- Uses explicit RTL `DSP48E2` primitive instances. Vivado resolves `DSP48E2`
  from its built-in UNISIM library.
- Creates a Vivado IP Catalog `DSP48 Macro` core
  (`xilinx.com:ip:xbip_dsp48_macro`) named `dsp48_macro_dspe2` so a DSP IP
  appears under **IP Sources** in the GUI.

## DSP48E2 RTL And IP Catalog

The actual arithmetic RTL uses manual `DSP48E2` primitive instantiation. The
`DSP48 Macro` IP (`xbip_dsp48_macro`) created by the script is added for
project/IP-catalog visibility in the GUI; it is not instantiated by
`deep_snn_top`.

Because the RTL primitives are not IP instances:

- RTL `DSP48E2` blocks appear after RTL elaboration or synthesis.
- The `dsp48_macro_dspe2` IP appears under **IP Sources**.
- Vivado simulation uses the `SIM` define from `sim_1`, which selects
  behavioral DSP math models instead of elaborating raw `DSP48E2` primitives.
- The `INFO: [IP_Flow ...]` messages during `create_project` are normal Vivado
  project startup messages.

To check DSP cells from the Vivado Tcl Console after elaboration or synthesis:

```tcl
get_cells -hier -filter {REF_NAME =~ DSP48E2*}
```

To print only the count:

```tcl
llength [get_cells -hier -filter {REF_NAME =~ DSP48E2*}]
```

The copy step is intentional. Vivado 2018.2 had trouble adding source files from
paths with spaces and parentheses, such as `SNN Parameters/Final Parameters (RTL)`.
Edit the original RTL files under `rtl/`, then rerun the TCL script to refresh
the staged copies.

## Simulation DSP Mode

The RTL DSP wrapper files contain `ifdef SIM` behavioral models. Use this define
when running a non-synthesis simulation so tools do not need to elaborate the
Xilinx `DSP48E2` primitive.

This project script applies the define only to the Vivado simulation fileset:

```tcl
set_property verilog_define {SIM} [get_filesets sim_1]
set_property -name xsim.elaborate.xelab.more_options -value {--timescale 1ns/1ps} -objects [get_filesets sim_1]
```

For command-line simulators, pass the equivalent compile option, for example:

```text
+define+SIM
xelab --timescale 1ns/1ps
```

Do not define `SIM` for synthesis. Without `SIM`, the RTL instantiates the real
manual `DSP48E2` blocks.

## Run From Vivado GUI

1. Open Vivado 2018.2.
2. In the Tcl Console, create the project without launching synthesis:

   ```tcl
   set argv [list -no-synth]
   source {C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl}
   ```

3. Open the generated project:

   ```tcl
   open_project {C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/build/vivado/deep_snn_top/deep_snn_top.xpr}
   ```

4. In Flow Navigator, click **Run Synthesis**.

If Vivado says the project is locked, close any old Vivado windows or running
Vivado processes that are using this project, then run the script again.

## Run Synthesis Directly From GUI Tcl Console

To create the project and launch `synth_1` from the Tcl Console:

```tcl
set argv [list]
source {C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl}
```

When synthesis completes, the script opens `synth_1` and prints a short DSP48E2
cell summary in the Tcl Console. It does not write timing or utilization reports.

## Quick RTL Elaboration Check

To create the project and run Vivado's synthesis front-end elaboration without
waiting for a full synthesis run:

```tcl
set argv [list -elab-only -keep-open]
source {C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl}
```

This is useful after RTL edits because it catches primitive, port, parameter,
and top integration errors faster than a full `synth_1` run. The `-keep-open`
option leaves the elaborated RTL design open in the GUI so you can inspect the
hierarchy and run `get_cells` queries.

## Run From Windows Command Prompt Or PowerShell

Create project only:

```powershell
& 'C:/Xilinx/Vivado/2018.2/bin/vivado.bat' -mode batch -source 'C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl' -tclargs -no-synth
```

Create project and run the quick elaboration check:

```powershell
& 'C:/Xilinx/Vivado/2018.2/bin/vivado.bat' -mode batch -source 'C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl' -tclargs -elab-only
```

Create project and run synthesis:

```powershell
& 'C:/Xilinx/Vivado/2018.2/bin/vivado.bat' -mode batch -source 'C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl'
```

## Notes

- The generated project uses staged copies under `build/vivado/src`.
- Do not edit the staged files directly; edit files under `rtl/`.
- Rerun the TCL script after RTL changes so Vivado sees the updated source set.
- The script currently builds only the incomplete top datapath, not the future
  controller, BRAM integration, or classifier head.
