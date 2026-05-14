## Classifier Head — FC Layers

Fully-connected layer hardware for the MultiStage-DeepSNN accelerator.

### Files
| File | What it does |
|------|-------------|
| `fc1_layer.sv` | FC1: 128 inputs -> 256 outputs + ReLU, 8 parallel MACs |
| `fc2_layer.sv` | FC2: 256 inputs -> 4 class logits, 4 parallel MACs |
| `tb_fc1_layer.sv` | Testbench for FC1 (3 test cases, self-checking) |
| `tb_fc2_layer.sv` | Testbench for FC2 (3 test cases, self-checking) |
| `golden_model.py` | Python golden model — computes expected fixed-point outputs |

### Dependencies
Weights and biases are pulled from the SNN parameter files in
`../SNN Parameters/Values For LUTS Before Optimization/`. Make sure those
files are present and the build system can resolve the `include paths.

### How to run testbenches (QuestaSim)
Open a terminal in this directory, then:

```
# FC1 (128->256 + ReLU, 4,096 cycles)
vlog -64 -sv -mfcu -suppress 2184 -work work fc1_layer.sv tb_fc1_layer.sv "+incdir+."
vsim -c tb_fc1_layer -do "run -all; quit"

# FC2 (256->4, 256 cycles)
vlog -64 -sv -mfcu -suppress 2184 -work work fc2_layer.sv tb_fc2_layer.sv "+incdir+."
vsim -c tb_fc2_layer -do "run -all; quit"
```

### Golden model
```
python golden_model.py --fc1 --random-seed 42 --num-cases 3
```
Prints SystemVerilog-formatted test vectors that can be pasted into the
testbench files.
