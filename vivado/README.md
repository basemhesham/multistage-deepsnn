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
  from its built-in UNISIM library, so no DSP IP core or extra IP repository is
  added by the script.

## Why DSP Blocks Do Not Appear Under IP Sources

This project uses manual `DSP48E2` primitive instantiation in RTL. That is
different from adding a Vivado "DSP48 Macro" IP core.

Because of that:

- `DSP48E2` blocks will not appear under **IP Sources**.
- The `INFO: [IP_Flow ...]` messages during `create_project` are normal Vivado
  project startup messages.
- `DSP48E2` cells appear only after RTL elaboration or synthesis.
- After synthesis, the script writes a Vivado-2018.2-compatible DSP48E2 cell
  report in:

  ```text
  build/vivado/reports/deep_snn_top_dsp_synth.rpt
  ```

  The normal utilization report also includes DSP usage:

  ```text
  build/vivado/reports/deep_snn_top_utilization_synth.rpt
  ```

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

When synthesis completes, reports are written to:

```text
build/vivado/reports/
```

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
