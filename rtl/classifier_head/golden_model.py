# ////////////////////////////////////////////////////////////////////////////////////////////////
# Author    : Ahmad Khattab
# Date      : 5/13/26
# File      : golden_model.py
# Status    : finalized
# Goal      : Fixed-point golden model for FC1 and FC2 layers.
#             Reads weight/bias parameter files, computes expected outputs
#             using the same Q7.10 arithmetic as the RTL hardware.
#
# Usage:
#     python golden_model.py                         # both layers, 5 test cases
#     python golden_model.py --fc1 --num-cases 10    # FC1 with 10 cases
#     python golden_model.py --fc2 --random-seed 42  # FC2 with custom seed
# ////////////////////////////////////////////////////////////////////////////////////////////////

import sys, os, re, random, argparse

DATA_WIDTH  = 18
FRAC_BITS   = 10
ACCUM_WIDTH = 48

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(SCRIPT_DIR)
PARAM_DIR   = os.path.join(REPO_ROOT, "SNN Parameters", "Values For LUTS Before Optimization")


def parse_sv_binary_array(filepath):
    """Parse 18'hXXXX / 18'bXXXX values from a SystemVerilog localparam file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    matches = re.findall(r"18'[bh]([01a-fA-FxXzZ_]+)", text)
    values = []
    for m in matches:
        clean = m.replace('_', '')
        val = int(clean, 2) if all(c in '01_' for c in m) else int(clean, 16)
        if val >= (1 << (DATA_WIDTH - 1)):
            val -= (1 << DATA_WIDTH)
        values.append(val)
    return values


def load_fc_weights_and_biases(fc_name):
    """Return (weights[outputs][inputs], biases[outputs], n_inputs, n_outputs)."""
    if fc_name == "fc1":
        w_file = os.path.join(PARAM_DIR, "FC1_WEIGHTS.sv")
        b_file = os.path.join(PARAM_DIR, "FC1_BIAS.sv")
        n_inputs, n_outputs = 128, 256
    else:
        w_file = os.path.join(PARAM_DIR, "FC2_WEIGHTS.sv")
        b_file = os.path.join(PARAM_DIR, "FC2_BIAS.sv")
        n_inputs, n_outputs = 256, 4

    w_flat = parse_sv_binary_array(w_file)
    biases  = parse_sv_binary_array(b_file)
    weights = [[w_flat[o * n_inputs + i] for i in range(n_inputs)] for o in range(n_outputs)]
    assert len(biases) == n_outputs
    return weights, biases, n_inputs, n_outputs


def fc_forward(inputs, weights, biases, n_inputs, n_outputs, apply_relu=True):
    """Compute FC layer in fixed-point, matching the RTL exactly."""
    outputs = []
    for o in range(n_outputs):
        acc = 0
        for i in range(n_inputs):
            acc += inputs[i] * weights[o][i]
        total = acc + (biases[o] << FRAC_BITS)
        result = total >> FRAC_BITS
        if apply_relu and result < 0:
            result = 0
        max_val = (1 << (DATA_WIDTH - 1)) - 1
        min_val = -(1 << (DATA_WIDTH - 1))
        if result > max_val:
            result = max_val
        elif result < min_val:
            result = min_val
        outputs.append(result)
    return outputs


def format_sv_array(values, bits=18):
    """Format a list of signed ints as SystemVerilog array literal entries."""
    out = []
    for v in values:
        uval = v & ((1 << bits) - 1)
        out.append(f"        18'h{uval:05X}")
    return ",\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="FC Golden Model for RTL Verification")
    parser.add_argument("--fc1", action="store_true")
    parser.add_argument("--fc2", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1)
    parser.add_argument("--num-cases", type=int, default=5)
    args = parser.parse_args()

    do_fc1 = args.fc1 or (not args.fc1 and not args.fc2)
    do_fc2 = args.fc2 or (not args.fc1 and not args.fc2)

    for fc in (['fc1'] if do_fc1 else []) + (['fc2'] if do_fc2 else []):
        w, b, ni, no = load_fc_weights_and_biases(fc)
        relu = (fc == 'fc1')
        rng = random.Random(args.random_seed)
        for case in range(args.num_cases):
            inp = [rng.randint(0, 1024) for _ in range(ni)]
            exp = fc_forward(inp, w, b, ni, no, relu)
            print(f"\n// ---- {fc.upper()} Test Case {case} ----")
            print(f"localparam logic signed [17:0] tb_in_{case} [{ni}] = '{{")
            print(format_sv_array(inp))
            print(f" }};")
            print(f"localparam logic signed [17:0] tb_exp_{case} [{no}] = '{{")
            print(format_sv_array(exp))
            print(f" }};")


if __name__ == "__main__":
    main()
