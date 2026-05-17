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

## Run From Windows Command Prompt Or PowerShell

Create project only:

```powershell
& 'C:/Xilinx/Vivado/2018.2/bin/vivado.bat' -mode batch -source 'C:/Users/PC/Desktop/STM/Repo/MultiStage-DeepSNN/vivado/create_deep_snn_top_project.tcl' -tclargs -no-synth
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
