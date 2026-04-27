import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
from numba import njit, prange
import csv 
import time
import os
import shutil
import random
LABEL_MAP = {
    "negative_samples": 0,        # no incident
    "drifting_or_skidding": 1,     # risky but no impact
    "other_crash": 2,              # crash, ego NOT involved
    "collision": 3,                # crash, ego involved (most severe)
}

# ----------------------------------------------------------------------
#  Silence harmless NNPACK warning
# ----------------------------------------------------------------------
torch.backends.nnpack.enabled = False

# ----------------------------------------------------------------------
#  Numba: scalar quantizer (division‑based, manual clamp)
# ----------------------------------------------------------------------
@njit
def quantize_scalar(v, scale, min_val, max_val):
    # Convert to fixed‑point integer, round, clamp, and scale back
    q = np.round(v * scale)
    # Clamp to representable integer range
    int_min = min_val * scale   # because min_val = int_min / scale
    int_max = max_val * scale
    if q < int_min:
        q = int_min
    if q > int_max:
        q = int_max
    return q / scale

# ----------------------------------------------------------------------
#  Numba: quantize entire array (elementwise)
# ----------------------------------------------------------------------


@njit(parallel=True)
def quantize_array(arr, scale, min_val, max_val):
    out = np.zeros_like(arr)
    int_min = min_val * scale
    int_max = max_val * scale
    for idx in np.ndindex(arr.shape):
        q = np.round(arr[idx] * scale)
        if q < int_min:
            q = int_min
        if q > int_max:
            q = int_max
        out[idx] = q / scale
    return out

# ----------------------------------------------------------------------
#  1. Batch Normalization (affine: A*x + B) – channel‑last
# ----------------------------------------------------------------------
@njit(parallel=True)
def myBatchNorm_numba(x, A, B, quant=False, scale=1.0, min_val=0, max_val=255):
    H, W, C = x.shape
    out = np.zeros_like(x)
    for c in prange(C):
        for h in range(H):
            for w in range(W):
                v = A[c] * x[h, w, c]
                
                if quant:
                    v = quantize_scalar(v, scale, min_val, max_val)
                v += B[c]
                if quant:
                    v = quantize_scalar(v, scale, min_val, max_val)
                out[h, w, c] = v
    return out

# ----------------------------------------------------------------------
#  2. Max Pooling 2x2, stride=2 – channel‑last
# ----------------------------------------------------------------------
@njit(parallel=True)
def myMaxPool2D_numba(x):
    H, W, C = x.shape
    Hout = H // 2
    Wout = W // 2
    out = np.zeros((Hout, Wout, C), dtype=x.dtype)
    for c in prange(C):
        for i in range(Hout):
            for j in range(Wout):
                m = x[2*i, 2*j, c]
                v2 = x[2*i+1, 2*j, c]
                v3 = x[2*i, 2*j+1, c]
                v4 = x[2*i+1, 2*j+1, c]
                if v2 > m: m = v2
                if v3 > m: m = v3
                if v4 > m: m = v4
                out[i, j, c] = m
    return out

# ----------------------------------------------------------------------
#  3. Convolution (stride, padding) – weight: (kH, kW, Cin, Cout)
# ----------------------------------------------------------------------
@njit(parallel=True)
def myConv2D_numba(x, weight, bias, stride=1, padding=0,
                   quant=False, scale=1.0, min_val=0, max_val=255):
    H, W, C_in = x.shape
    kH, kW, _, C_out = weight.shape

    if padding > 0:
        x_pad = np.zeros((H+2*padding, W+2*padding, C_in), dtype=x.dtype)
        x_pad[padding:H+padding, padding:W+padding, :] = x
    else:
        x_pad = x

    Hp, Wp = x_pad.shape[0], x_pad.shape[1]
    H_out = (Hp - kH) // stride + 1
    W_out = (Wp - kW) // stride + 1
    out = np.zeros((H_out, W_out, C_out), dtype=x.dtype)

    for co in prange(C_out):
        for ci in range(C_in):
            for kh in range(kH):
                for kw in range(kW):
                    w_val = weight[kh, kw, ci, co]
                    for h in range(H_out):
                        h_in = h * stride + kh
                        for w in range(W_out):
                            w_in = w * stride + kw
                            x= x_pad[h_in, w_in, ci] * w_val
                            if quant:
                                x = quantize_scalar(x, scale, min_val, max_val)

                            out[h, w, co] +=x
                            if quant:
                                out[h, w, co] = quantize_scalar(out[h, w, co], scale, min_val, max_val)


    for co in prange(C_out):
        bias_val = bias[co]
        for h in range(H_out):
            for w in range(W_out):
                val = out[h, w, co] + bias_val
                if quant:
                    val = quantize_scalar(val, scale, min_val, max_val)
                out[h, w, co] = val
    return out

# # ----------------------------------------------------------------------
# #  4. LIF Neuron (one timestep, hard reset, reset_delay=True)
# # ----------------------------------------------------------------------
# @njit(parallel=True)
# def myLIF_numba_quant(x, mem, beta, threshold,
#                       quant, scale, min_val, max_val):
#     H, W, C = x.shape
#     spk = np.zeros_like(mem)
#     new_mem = np.zeros_like(mem)

#     # Ensure constants are float32 for consistency
#     beta = np.float32(beta)
#     threshold = np.float32(threshold)
#     scale = np.float32(scale)
#     min_val = np.float32(min_val)
#     max_val = np.float32(max_val)

#     for i in prange(H):
#         for j in range(W):
#             for k in range(C):
#                 # --- Step 1: compute beta * mem ---
#                 prod = beta * mem[i, j, k]
#                 if quant:
#                     prod = quantize_scalar(prod, scale, min_val, max_val)

#                 # --- Step 2: add input x ---
#                 sum1 = prod + x[i, j, k]
#                 if quant:
#                     sum1 = quantize_scalar(sum1, scale, min_val, max_val)

#                 # --- Step 3: compute reset term ---
#                 reset = 1.0 if mem[i, j, k] >= threshold else 0.0
#                 reset_term = reset * threshold   # either 0 or threshold
#                 if quant:
#                     reset_term = quantize_scalar(reset_term, scale, min_val, max_val)

#                 # --- Step 4: subtract reset term ---
#                 m = sum1 - reset_term
#                 if quant:
#                     m = quantize_scalar(m, scale, min_val, max_val)

#                 # --- Step 5: spike generation ---
#                 s = 1.0 if m >= threshold else 0.0
#                 if quant:
#                     s = quantize_scalar(s, scale, min_val, max_val)

#                 spk[i, j, k] = s
#                 new_mem[i, j, k] = m
#     return spk, new_mem


# ----------------------------------------------------------------------
#  4. LIF Neuron (one timestep, hard reset, reset_delay=True)
# ----------------------------------------------------------------------
@njit(parallel=True)
def myLIF_numba_quant(spk_in,x, mem, beta, threshold,
                      quant, scale, min_val, max_val):
    H, W, C = x.shape
    spk_out = np.zeros_like(mem)
    new_mem = np.zeros_like(mem)

    # Ensure constants are float32 for consistency
    beta = np.float32(beta)
    threshold = np.float32(threshold)
    scale = np.float32(scale)
    min_val = np.float32(min_val)
    max_val = np.float32(max_val)

    for i in prange(H):
        for j in range(W):
            for k in range(C):
                # --- Step 1: compute beta * mem ---
                prod = beta * mem[i, j, k]
                if quant:
                    prod = quantize_scalar(prod, scale, min_val, max_val)

                # --- Step 2: add input x ---
                sum1 = prod + x[i, j, k]
                if quant:
                    sum1 = quantize_scalar(sum1, scale, min_val, max_val)

                reset_term = spk_in[i,j,k] * threshold   # either 0 or threshold
                if quant:
                    reset_term = quantize_scalar(reset_term, scale, min_val, max_val)

                # --- Step 4: subtract reset term ---
                m = sum1 - reset_term
                if quant:
                    m = quantize_scalar(m, scale, min_val, max_val)

                # --- Step 5: spike generation ---
                s = 1.0 if m >= threshold else 0.0
                if quant:
                    s = quantize_scalar(s, scale, min_val, max_val)

                spk_out[i, j, k] = s
                new_mem[i, j, k] = m
    return spk_out, new_mem
# ----------------------------------------------------------------------
#  5. Global Average Pooling – channel‑last
# ----------------------------------------------------------------------
@njit(parallel=True)
def myGAP_numba(x, quant=False, scale=1.0, min_val=0, max_val=255):
    H, W, C = x.shape
    out = np.zeros(C, dtype=np.float32)
    for c in prange(C):
        acc = 0.0
        for h in range(H):
            for w in range(W):
                acc += x[h, w, c]
                # if quant:
                #     acc = quantize_scalar(acc, scale, min_val, max_val)
        acc /= (H * W)
        if quant:
            acc = quantize_scalar(acc, scale, min_val, max_val)
        out[c] = acc
    return out

# ----------------------------------------------------------------------
#  6. Dense (Fully Connected) – optional ReLU, per‑product quant
# ----------------------------------------------------------------------
@njit(parallel=True)
def myDense_numba(x, W, b, relu=False,
                  quant=False, scale=1.0, min_val=0, max_val=255):
    Nin, Nout = W.shape
    out = np.zeros(Nout, dtype=np.float32)
    for o in prange(Nout):
        acc = 0.0
        for i in range(Nin):
            val = x[i] * W[i, o]
            if quant:
                val = quantize_scalar(val, scale, min_val, max_val)
            acc += val
            if quant:
                acc = quantize_scalar(acc, scale, min_val, max_val)
        acc += b[o]
        if relu and acc < 0:
            acc = 0.0
        if quant:
            acc = quantize_scalar(acc, scale, min_val, max_val)
        out[o] = acc
    return out

# ======================================================================
#  PyTorch Model (Reference) – exactly as in your training
# ======================================================================
class DeepSNNClassification_g11(nn.Module):
    def __init__(self, num_classes, beta=0.9, spike_grad=surrogate.atan(), dropout_prob=0.4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (B, T, H, W, C) -> (B, T, C, H, W)
        x = x.permute(0, 1, 4, 2, 3)

        self.lif1.reset_mem()
        self.lif2.reset_mem()
        self.lif3.reset_mem()

        B, T, _, _, _ = x.shape
        mem1 = mem2 = mem3 = None

        for t in range(T):
            xt = x[:, t]
            out = self.pool1(self.bn1(self.conv1(xt)))
            spk1, mem1 = self.lif1(out, mem1)

            out = self.pool2(self.bn2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(out, mem2)

            out = self.pool3(self.bn3(self.conv3(spk2)))
            spk3, mem3 = self.lif3(out, mem3)

        out = self.global_pool(spk3)
        out = out.reshape(B, -1)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# ======================================================================
#  Hardware Model Wrapper (Numba) – WITH CORRECT LIF PARAMETERS
# ======================================================================
class CustomQuantizedG11:
    def __init__(self, pt_model, quantization=True, total_bits=16, frac_bits=10):
        self.quant = quantization
        self.scale = 2**frac_bits
        self.min_val = -(2**(total_bits-1)) / self.scale
        self.max_val = (2**(total_bits-1)-1) / self.scale

        # --- Extract trained LIF parameters ---
        self.beta1 = pt_model.lif1.beta.detach().cpu().numpy().item()
        self.beta2 = pt_model.lif2.beta.detach().cpu().numpy().item()
        self.beta3 = pt_model.lif3.beta.detach().cpu().numpy().item()
        self.thresh1 = pt_model.lif1.threshold.detach().cpu().numpy().item()
        self.thresh2 = pt_model.lif2.threshold.detach().cpu().numpy().item()
        self.thresh3 = pt_model.lif3.threshold.detach().cpu().numpy().item()

        # Convolution parameters
        self.pad1 = pt_model.conv1.padding[0]
        self.pad2 = pt_model.conv2.padding[0]
        self.pad3 = pt_model.conv3.padding[0]
        self.stride1 = pt_model.conv1.stride[0]
        self.stride2 = pt_model.conv2.stride[0]
        self.stride3 = pt_model.conv3.stride[0]

        def get_wb(conv):
            w = conv.weight.data.cpu().numpy().transpose(2,3,1,0)  # (Kh,Kw,Cin,Cout)
            b = conv.bias.data.cpu().numpy()
            return w, b

        w1, b1 = get_wb(pt_model.conv1)
        w2, b2 = get_wb(pt_model.conv2)
        w3, b3 = get_wb(pt_model.conv3)

        def fuse_bn(bn):
            g = bn.weight.data.cpu().numpy()
            b = bn.bias.data.cpu().numpy()
            m = bn.running_mean.data.cpu().numpy()
            v = bn.running_var.data.cpu().numpy()
            eps = 1e-5
            A = g / np.sqrt(v + eps)
            B = b - g * m / np.sqrt(v + eps)
            return A, B

        self.bn1_A, self.bn1_B = fuse_bn(pt_model.bn1)
        self.bn2_A, self.bn2_B = fuse_bn(pt_model.bn2)
        self.bn3_A, self.bn3_B = fuse_bn(pt_model.bn3)

        self.w_fc1 = pt_model.fc1.weight.data.cpu().numpy().T
        self.b_fc1 = pt_model.fc1.bias.data.cpu().numpy()
        self.w_fc2 = pt_model.fc2.weight.data.cpu().numpy().T
        self.b_fc2 = pt_model.fc2.bias.data.cpu().numpy()

        if self.quant:
            self.w1 = quantize_array(w1, self.scale, self.min_val, self.max_val)
            self.b1 = quantize_array(b1, self.scale, self.min_val, self.max_val)
            self.w2 = quantize_array(w2, self.scale, self.min_val, self.max_val)
            self.b2 = quantize_array(b2, self.scale, self.min_val, self.max_val)
            self.w3 = quantize_array(w3, self.scale, self.min_val, self.max_val)
            self.b3 = quantize_array(b3, self.scale, self.min_val, self.max_val)
            self.bn1_A = quantize_array(self.bn1_A, self.scale, self.min_val, self.max_val)
            self.bn1_B = quantize_array(self.bn1_B, self.scale, self.min_val, self.max_val)
            self.bn2_A = quantize_array(self.bn2_A, self.scale, self.min_val, self.max_val)
            self.bn2_B = quantize_array(self.bn2_B, self.scale, self.min_val, self.max_val)
            self.bn3_A = quantize_array(self.bn3_A, self.scale, self.min_val, self.max_val)
            self.bn3_B = quantize_array(self.bn3_B, self.scale, self.min_val, self.max_val)
            self.w_fc1 = quantize_array(self.w_fc1, self.scale, self.min_val, self.max_val)
            self.b_fc1 = quantize_array(self.b_fc1, self.scale, self.min_val, self.max_val)
            self.w_fc2 = quantize_array(self.w_fc2, self.scale, self.min_val, self.max_val)
            self.b_fc2 = quantize_array(self.b_fc2, self.scale, self.min_val, self.max_val)
        else:
            self.w1, self.b1 = w1, b1
            self.w2, self.b2 = w2, b2
            self.w3, self.b3 = w3, b3

    def forward(self, x, return_intermediates=False):
        # x: (1, T, H, W) or (1, T, H, W, C) – batch size must be 1
        x = x[0]                # (T, H, W) or (T, H, W, C)
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)  # (T, H, W, 1)
        T, H, W, C = x.shape

        # Initialise membrane potentials (channel‑last)
        mem1 = np.zeros((H//2, W//2, 32), dtype=np.float32)
        mem2 = np.zeros((H//4, W//4, 64), dtype=np.float32)
        mem3 = np.zeros((H//8, W//8, 128), dtype=np.float32)
        spk1 = np.zeros((H//2, W//2, 32), dtype=np.float32)
        spk2 = np.zeros((H//4, W//4, 64), dtype=np.float32)
        spk3 = np.zeros((H//8, W//8, 128), dtype=np.float32)
        intermediates = {} if return_intermediates else None

        for t in range(T):
            xt = x[t]
            if self.quant:
                xt = quantize_array(xt, self.scale, self.min_val, self.max_val)

            # ---- Block 1 ----
            out = myConv2D_numba(xt, self.w1, self.b1,
                                 stride=self.stride1, padding=self.pad1,
                                 quant=self.quant, scale=self.scale,
                                 min_val=self.min_val, max_val=self.max_val)
            if return_intermediates: intermediates['conv1'] = out.copy()
            out = myBatchNorm_numba(out, self.bn1_A, self.bn1_B,
                                    self.quant, self.scale, self.min_val, self.max_val)
            if return_intermediates: intermediates['bn1'] = out.copy()
            out = myMaxPool2D_numba(out)
            if return_intermediates: intermediates['pool1'] = out.copy()
            spk1, mem1 = myLIF_numba_quant(spk1,out, mem1,
                                           beta=self.beta1, threshold=self.thresh1,
                                           quant=self.quant, scale=self.scale,
                                           min_val=self.min_val, max_val=self.max_val)
            if return_intermediates:
                intermediates['lif1_spk'] = spk1.copy()
                intermediates['lif1_mem'] = mem1.copy()

            # ---- Block 2 ----
            out = myConv2D_numba(spk1, self.w2, self.b2,
                                 stride=self.stride2, padding=self.pad2,
                                 quant=self.quant, scale=self.scale,
                                 min_val=self.min_val, max_val=self.max_val)
            if return_intermediates: intermediates['conv2'] = out.copy()
            out = myBatchNorm_numba(out, self.bn2_A, self.bn2_B,
                                    self.quant, self.scale, self.min_val, self.max_val)
            if return_intermediates: intermediates['bn2'] = out.copy()
            out = myMaxPool2D_numba(out)
            if return_intermediates: intermediates['pool2'] = out.copy()
            spk2, mem2 = myLIF_numba_quant(spk2,out, mem2,
                                           beta=self.beta2, threshold=self.thresh2,
                                           quant=self.quant, scale=self.scale,
                                           min_val=self.min_val, max_val=self.max_val)
            if return_intermediates:
                intermediates['lif2_spk'] = spk2.copy()
                intermediates['lif2_mem'] = mem2.copy()

            # ---- Block 3 ----
            out = myConv2D_numba(spk2, self.w3, self.b3,
                                 stride=self.stride3, padding=self.pad3,
                                 quant=self.quant, scale=self.scale,
                                 min_val=self.min_val, max_val=self.max_val)
            if return_intermediates: intermediates['conv3'] = out.copy()
            out = myBatchNorm_numba(out, self.bn3_A, self.bn3_B,
                                    self.quant, self.scale, self.min_val, self.max_val)
            if return_intermediates: intermediates['bn3'] = out.copy()
            out = myMaxPool2D_numba(out)
            if return_intermediates: intermediates['pool3'] = out.copy()
            spk3, mem3 = myLIF_numba_quant(spk3,out, mem3,
                                           beta=self.beta3, threshold=self.thresh3,
                                           quant=self.quant, scale=self.scale,
                                           min_val=self.min_val, max_val=self.max_val)
            if return_intermediates:
                intermediates['lif3_spk'] = spk3.copy()
                intermediates['lif3_mem'] = mem3.copy()

        # ---- Global Average Pooling ----
        out = myGAP_numba(spk3, self.quant, self.scale, self.min_val, self.max_val)
        if return_intermediates: intermediates['gap'] = out.copy()

        # ---- FC1 + ReLU ----
        out = myDense_numba(out, self.w_fc1, self.b_fc1, relu=True,
                            quant=self.quant, scale=self.scale,
                            min_val=self.min_val, max_val=self.max_val)
        if return_intermediates: intermediates['fc1'] = out.copy()

        # ---- FC2 (no ReLU) ----
        out = myDense_numba(out, self.w_fc2, self.b_fc2, relu=False,
                            quant=self.quant, scale=self.scale,
                            min_val=self.min_val, max_val=self.max_val)
        if return_intermediates: intermediates['fc2'] = out.copy()

        if return_intermediates:
            return out, intermediates
        return out

# ======================================================================
#  PyTorch intermediate capture (channel‑first)
# ======================================================================
def get_pt_intermediates(model, x_batch):
    model.eval()
    x = x_batch.float().to(next(model.parameters()).device)
    B, T, H, W, C = x.shape
    x = x.permute(0, 1, 4, 2, 3)

    model.lif1.reset_mem()
    model.lif2.reset_mem()
    model.lif3.reset_mem()
    mem1 = mem2 = mem3 = None
    intermediates = {}

    for t in range(T):
        xt = x[:, t]

        out = model.conv1(xt)
        intermediates['conv1'] = out[0].detach().cpu().numpy()
        out = model.bn1(out)
        intermediates['bn1'] = out[0].detach().cpu().numpy()
        out = model.pool1(out)
        intermediates['pool1'] = out[0].detach().cpu().numpy()
        spk1, mem1 = model.lif1(out, mem1)
        intermediates['lif1_spk'] = spk1[0].detach().cpu().numpy()
        intermediates['lif1_mem'] = mem1[0].detach().cpu().numpy()

        out = model.conv2(spk1)
        intermediates['conv2'] = out[0].detach().cpu().numpy()
        out = model.bn2(out)
        intermediates['bn2'] = out[0].detach().cpu().numpy()
        out = model.pool2(out)
        intermediates['pool2'] = out[0].detach().cpu().numpy()
        spk2, mem2 = model.lif2(out, mem2)
        intermediates['lif2_spk'] = spk2[0].detach().cpu().numpy()
        intermediates['lif2_mem'] = mem2[0].detach().cpu().numpy()

        out = model.conv3(spk2)
        intermediates['conv3'] = out[0].detach().cpu().numpy()
        out = model.bn3(out)
        intermediates['bn3'] = out[0].detach().cpu().numpy()
        out = model.pool3(out)
        intermediates['pool3'] = out[0].detach().cpu().numpy()
        spk3, mem3 = model.lif3(out, mem3)
        intermediates['lif3_spk'] = spk3[0].detach().cpu().numpy()
        intermediates['lif3_mem'] = mem3[0].detach().cpu().numpy()

    out = model.global_pool(spk3).reshape(B, -1)
    intermediates['gap'] = out[0].detach().cpu().numpy()
    out = model.fc1(out)
    intermediates['fc1'] = out[0].detach().cpu().numpy()
    out = F.relu(out)
    intermediates['fc1_relu'] = out[0].detach().cpu().numpy()
    out = model.dropout(out)   # eval → identity
    out = model.fc2(out)
    intermediates['fc2'] = out[0].detach().cpu().numpy()
    return intermediates

# ======================================================================
#  Enhanced Deep Comparison (single sample) with spike mismatch counting
# ======================================================================
def deep_compare(pt_model, hw_model, sample_batch,
                 rtol=1e-3, atol=1e-3,
                 spike_mismatch_threshold=0.02):
    print("\n🔬 DEEP COMPARISON (quantization OFF) – single sample")
    print("=" * 70)
    x_np = sample_batch['video'].float().cpu().numpy()
    pt_ints = get_pt_intermediates(pt_model, sample_batch['video'])
    hw_out, hw_ints = hw_model.forward(x_np, return_intermediates=True)

    all_match = True
    spike_mismatch_rates = {}

    for key in hw_ints.keys():
        if key not in pt_ints:
            continue

        pt_val = pt_ints[key]
        hw_val = hw_ints[key]

        # Convert HW layout (H,W,C) -> (C,H,W) for comparison
        if hw_val.ndim == 3:
            hw_val_perm = hw_val.transpose(2, 0, 1)
        else:
            hw_val_perm = hw_val

        if '_spk' in key:
            mismatches = np.sum(pt_val != hw_val_perm)
            total_pixels = pt_val.size
            mismatch_rate = mismatches / total_pixels
            spike_mismatch_rates[key] = (mismatches, total_pixels, mismatch_rate)

            if mismatches == 0:
                print(f"✅ {key:15s} : perfect match (0 mismatches)")
            else:
                print(f"❌ {key:15s} : {mismatches:5d} / {total_pixels:6d} spikes differ "
                      f"({mismatch_rate:.4%})")
                if mismatch_rate <= spike_mismatch_threshold:
                    print(f"   (✓ below threshold {spike_mismatch_threshold:.2%})")
                else:
                    all_match = False
        else:
            try:
                np.testing.assert_allclose(pt_val, hw_val_perm, rtol=rtol, atol=atol)
                print(f"✅ {key:15s} : match")
            except AssertionError:
                max_diff = np.max(np.abs(pt_val - hw_val_perm))
                mean_diff = np.mean(np.abs(pt_val - hw_val_perm))
                print(f"❌ {key:15s} : mismatch (max diff={max_diff:.6f}, mean diff={mean_diff:.6f})")
                all_match = False

    pt_pred = np.argmax(pt_ints['fc2'])
    hw_pred = np.argmax(hw_out)
    print(f"\nPyTorch prediction : {pt_pred}")
    print(f"HW prediction      : {hw_pred}")
    if pt_pred == hw_pred:
        print("✅ Final predictions match!")
    else:
        print("❌ Final predictions differ!")
        all_match = False

    if all_match:
        print("\n🎉 All layers match within tolerance! HW model is correct (before quantization).")
    else:
        print("\n⚠️  Some mismatches – evaluating significance...")
        spike_ok = all(rate <= spike_mismatch_threshold
                      for _, _, rate in spike_mismatch_rates.values())
        if spike_ok and pt_pred == hw_pred:
            print(f"   ✅ But spike mismatch rate ≤ {spike_mismatch_threshold:.2%} "
                  "and predictions match → model is FUNCTIONALLY CORRECT.")
            all_match = True
        else:
            print("   ❌ Model not accepted – either predictions differ or spike mismatch exceeds threshold.")
    return all_match


# ======================================================================
#  DEEP COMPARISON ON SUBSET (many samples) – per‑layer statistics + final metrics
# ======================================================================
def deep_compare_subset(pt_model, hw_model, loader, num_samples=50,
                        rtol=1e-3, atol=1e-3,
                        spike_mismatch_threshold=0.001,strings_=True):
    """
    Run DEEP per‑layer comparison on a random subset of samples.
    Prints:
      - per‑layer spike mismatches and tensor differences
      - final prediction match rate
      - accuracy, macro F1, and classification report for both models
    Returns (passed, stats_dict, pt_metrics, hw_metrics).
    """
    all_batches = []
    for i, batch in enumerate(loader):
        if i >= 1000:
            break
        all_batches.append(batch)
    n_samples = min(num_samples, len(all_batches))
    sampled_batches = random.sample(all_batches, n_samples)

    print(f"\n🔬 DEEP COMPARISON ON {n_samples} RANDOM SAMPLES")
    print("=" * 80)

    stats = {}  # layer_name -> {'type': 'spike'/'tensor', 'diffs': [], 'mismatches': [], 'total_pixels': int}
    pt_preds = []
    hw_preds = []
    true_labels = []

    #for idx, batch in enumerate(sampled_batches):
    for idx, batch in enumerate(tqdm(sampled_batches, desc="Processing samples")):

        x_np = batch["video"].float().cpu().numpy()
        # Get PyTorch intermediates and final output
        pt_ints = get_pt_intermediates(pt_model, batch["video"])
        pt_out = pt_model(batch["video"].float().to(next(pt_model.parameters()).device))
        pt_pred = pt_out.argmax(1).item()

        # Get HW intermediates and final output
        hw_out, hw_ints = hw_model.forward(x_np, return_intermediates=True)
        hw_pred = np.argmax(hw_out)

        # Store predictions and true label
        if strings_:
           true_label= LABEL_MAP[batch["label"][0]]
        else:
           true_label = batch["label"][0].item()

        if isinstance(true_label, torch.Tensor):
            true_label = true_label.item()
        elif isinstance(true_label, (list, np.ndarray)) and len(true_label) == 1:
            true_label = true_label[0]
        true_labels.append(true_label)
        pt_preds.append(pt_pred)
        hw_preds.append(hw_pred)

        # Per‑layer comparison (same as before)
        for key in hw_ints.keys():
            if key not in pt_ints:
                continue

            pt_val = pt_ints[key]
            hw_val = hw_ints[key]

            if hw_val.ndim == 3:
                hw_val_perm = hw_val.transpose(2, 0, 1)
            else:
                hw_val_perm = hw_val

            if key not in stats:
                stats[key] = {
                    'type': 'spike' if '_spk' in key else 'tensor',
                    'diffs': [],
                    'mismatches': [],
                    'total_pixels': pt_val.size if '_spk' in key else None
                }

            if '_spk' in key:
                mismatches = np.sum(pt_val != hw_val_perm)
                stats[key]['mismatches'].append(mismatches)
            else:
                diff = np.abs(pt_val - hw_val_perm)
                stats[key]['diffs'].append({
                    'max': np.max(diff),
                    'mean': np.mean(diff)
                })

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{n_samples} samples...")

    # ------------------- Print per‑layer statistics -------------------
    print("\n" + "=" * 80)
    print(f"{'Layer':<20} {'Type':<10} {'Metric':<30} {'Value':<15}")
    print("=" * 80)

    all_layers_pass = True
    for layer, data in sorted(stats.items()):
        if data['type'] == 'spike':
            mismatches = np.array(data['mismatches'])
            total_pixels = data['total_pixels']
            avg_mismatches = mismatches.mean()
            max_mismatches = mismatches.max()
            avg_rate = avg_mismatches / total_pixels
            max_rate = max_mismatches / total_pixels

            pass_layer = avg_rate <= spike_mismatch_threshold
            if not pass_layer:
                all_layers_pass = False

            status = "✅ PASS" if pass_layer else "❌ FAIL"
            print(f"{layer:<20} {'spike':<10} {'Avg mismatches':<30} {avg_mismatches:.3f} / {total_pixels} ({avg_rate:.4%})")
            print(f"{'':<20} {'':<10} {'Max mismatches':<30} {max_mismatches} ({max_rate:.4%})")
            print(f"{'':<20} {'':<10} {'Threshold':<30} ≤ {spike_mismatch_threshold:.2%} {status}")
        else:
            diffs_max = [d['max'] for d in data['diffs']]
            diffs_mean = [d['mean'] for d in data['diffs']]

            avg_max = np.mean(diffs_max)
            max_max = np.max(diffs_max)
            avg_mean = np.mean(diffs_mean)
            max_mean = np.max(diffs_mean)

            print(f"{layer:<20} {'tensor':<10} {'Avg max diff':<30} {avg_max:.6f}")
            print(f"{'':<20} {'':<10} {'Max max diff':<30} {max_max:.6f}")
            print(f"{'':<20} {'':<10} {'Avg mean diff':<30} {avg_mean:.6f}")
            print(f"{'':<20} {'':<10} {'Max mean diff':<30} {max_mean:.6f}")

    # ------------------- Prediction agreement -------------------
    print("\n" + "=" * 80)
    print("FINAL PREDICTION AGREEMENT")
    pred_matches = sum(1 for pt, hw in zip(pt_preds, hw_preds) if pt == hw)
    pred_rate = pred_matches / n_samples
    print(f"  Prediction match rate: {pred_matches}/{n_samples} ({pred_rate:.2%})")
    pred_ok = pred_rate == 1.0

    # ------------------- Overall metrics for both models -------------------
    print("\n" + "=" * 80)
    print("CLASSIFICATION PERFORMANCE ON SUBSET")
    print("=" * 80)

    # PyTorch model
    pt_acc = accuracy_score(true_labels, pt_preds)
    pt_f1 = f1_score(true_labels, pt_preds, average="macro", zero_division=0)
    print("\n📈 PyTorch Model (float):")
    print(f"  Accuracy: {pt_acc:.4f}")
    print(f"  Macro F1 : {pt_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(true_labels, pt_preds, zero_division=0))

    # HW model (quant off)
    hw_acc = accuracy_score(true_labels, hw_preds)
    hw_f1 = f1_score(true_labels, hw_preds, average="macro", zero_division=0)
    print("\n📈 Hardware Model (quantization OFF):")
    print(f"  Accuracy: {hw_acc:.4f}")
    print(f"  Macro F1 : {hw_f1:.4f}")
    print("\n  Classification Report:")
    print(classification_report(true_labels, hw_preds, zero_division=0))

    # ------------------- Final verdict -------------------
    print("\n" + "=" * 80)
    if all_layers_pass and pred_ok:
        print("🎉 DEEP COMPARISON PASSED: All layers meet tolerance, predictions match 100%.")
        passed = True
    else:
        print("❌ DEEP COMPARISON FAILED:")
        if not all_layers_pass:
            print("   - Some spike layers exceed mismatch threshold.")
        if not pred_ok:
            print(f"   - Prediction match rate = {pred_rate:.2%} (must be 100%).")
        passed = False
    print("=" * 80)

    # Return metrics if needed (e.g., to avoid re‑running evaluations)
    pt_metrics = {'accuracy': pt_acc, 'f1': pt_f1}
    hw_metrics = {'accuracy': hw_acc, 'f1': hw_f1}
    return passed, stats, pt_metrics, hw_metrics

# ======================================================================
#  Evaluation functions (full test set) – robust label handling
# ======================================================================\


# ======================================================================
#  Evaluation functions (full test set) – robust label handling
# ======================================================================
@torch.no_grad()
def eval_float_model(model, loader, device, class_names=None,end_batch=None):
    y_true, y_pred = [], []
    batch_number = 0
    for batch in loader:
        x = batch["video"].to(device).float()
        # Extract label safely
        if end_batch is not None :
            y = batch["label"][0].item()
        else:
            y = LABEL_MAP[batch["label"][0]]
        
        out = model(x)
        p = out.argmax(1).item()
        y_true.append(y)
        y_pred.append(p)
        batch_number += 1
        if batch_number==end_batch:
            break

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("\nFLOAT MODEL EVAL COMPLETE")
    print(f"Acc = {acc:.4f}, F1 = {f1:.4f}")
    if class_names is not None:
        labels = sorted(set(y_true))
        target_names = [class_names[i] for i in labels]
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels,
                                   target_names=target_names, zero_division=0))
    return acc, f1

def eval_hw_model(hw_model, loader, class_names=None, duration_file="logs/hw_batch_durations.txt", verbose=True,end_batch=None):
    y_true, y_pred = [], []
    batch_number = 0
    seconds_total = 0
    total_batches = len(loader)

    os.makedirs(os.path.dirname(duration_file), exist_ok=True)
    if os.path.exists(duration_file):
        shutil.move(duration_file, f"{duration_file.rstrip('.txt')}_backup.txt")

    with open(duration_file, "a", buffering=1) as f:
        if os.path.getsize(duration_file) == 0:
            f.write("Batch_Number,Duration_Seconds,End_Time\n")

        for batch in loader:
            start = time.time()
            x = batch["video"].float().cpu().numpy()
            if end_batch is not None :
                y = batch["label"][0].item()
            else:
                y = LABEL_MAP[batch["label"][0]]
            out = hw_model.forward(x)
            p = np.argmax(out)
            y_true.append(y)
            y_pred.append(p)
            dur = time.time() - start
            seconds_total += dur
            if end_batch is not None:
                print(f"Batch {batch_number}/{end_batch} – Duration: {dur:.4f}s, Cumulative: {seconds_total:.2f}s")
            f.write(f"{batch_number},{dur:.4f},{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()
            
         
            batch_number += 1

            if verbose and batch_number % 50 == 0:
                print(f"    ... processed {batch_number}/{total_batches} batches")

            if batch_number == end_batch:
                break

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    print("\n--- HW EVALUATION COMPLETE ---")
    print(f"Total seconds = {seconds_total:.2f}")
    print(f"Accuracy = {acc:.4f}, F1 Score = {f1:.4f}")
    if class_names is not None:
        labels = sorted(set(y_true))
        target_names = [class_names[i] for i in labels]
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, labels=labels,
                                   target_names=target_names, zero_division=0))
    return acc, f1

# ======================================================================
#  MAIN – CONFIGURE BEFORE RUNNING
# ======================================================================
def ordi():
    # -------------------- CONFIGURE THESE --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/student_epoch_80_f1_0.7315_acc_0.7574_g11.pt"   # <-- YOUR CHECKPOINT
    NUM_CLASSES = 4
    CLASS_NAMES = ['negative_samples', 'drifting_or_skidding', 'other_crash', 'collision']   

    # -------------------- REPLACE WITH YOUR DATALOADER --------------------
    from kd import fourth_step
    _, test_dl = fourth_step(batch_size=1)
  

    # ---------- Load PyTorch model ----------
    pt_model = DeepSNNClassification_g11(num_classes=NUM_CLASSES).to(device)
    if os.path.exists(ckpt_path):
        pt_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        pt_model.eval()
        print("✅ Model loaded from checkpoint.")
    else:
        print(f"⚠️ Checkpoint not found – using untrained model.")

    # ---------- DEBUG: Verify spatial dimensions ----------
    print("\n🔍 VERIFYING MODEL SPATIAL DIMENSIONS:")
    dummy_input = torch.randn(1, 1, 256, 256).to(device)  # single frame, no time dim
    with torch.no_grad():
        out = pt_model.conv1(dummy_input)
        print(f"  After conv1: {out.shape}")
        out = pt_model.pool1(out)
        print(f"  After pool1: {out.shape}")
        out = pt_model.conv2(out)
        print(f"  After conv2: {out.shape}")
        out = pt_model.pool2(out)
        print(f"  After pool2: {out.shape}")
        out = pt_model.conv3(out)
        print(f"  After conv3: {out.shape}")
        out = pt_model.pool3(out)
        print(f"  After pool3: {out.shape}")
        out = pt_model.global_pool(out)
        print(f"  After global_pool: {out.shape}")
    print("    ✅ If spatial dimensions decrease as expected, input size is correct.")

    # ---------- 1. DEEP COMPARISON (Quant OFF) – single sample ----------
    sample_batch = next(iter(test_dl))
    hw_float = CustomQuantizedG11(pt_model, quantization=False)
    deep_compare(pt_model, hw_float, sample_batch, rtol=1e-3, atol=1e-3)

    # ---------- 2. DEEP COMPARISON ON 50 SAMPLES ----------
    passed, stats, pt_metrics, hw_metrics = deep_compare_subset(
        pt_model, hw_float, test_dl,
        num_samples=100,
        spike_mismatch_threshold=0.02,   # 0.1% spike mismatches allowed
        rtol=1e-3, atol=1e-3,strings_=False)

    # ---------- 3. FULL EVALUATION – FLOAT MODEL ----------
    print("\n" + "#" * 90)
    eval_float_model(pt_model, test_dl, device, CLASS_NAMES)
    print("#" * 90)

    # ---------- 4. FULL EVALUATION – HW (QUANT OFF) ----------
    start = time.time()
    eval_hw_model(hw_float, test_dl, CLASS_NAMES)
    print(f"HW (Quant OFF) total time: {time.time() - start:.2f}s")
    print("#" * 90)



def final():
    # -------------------- CONFIGURE THESE --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/student_epoch_80_f1_0.7315_acc_0.7574_g11.pt"
    NUM_CLASSES = 4
    CLASS_NAMES = ['negative_samples', 'drifting_or_skidding', 'other_crash', 'collision']
    SUBSET_SIZE = 20
    # Threshold for acceptable accuracy drop compared to float baseline (e.g., 98%)
    ACC_TOLERANCE = 0.98   # keep if quantized acc >= float_acc * ACC_TOLERANCE

    # -------------------- LOAD TEST DATALOADER --------------------
    from kd import fourth_step
    _, test_dl = fourth_step(batch_size=1)

    # -------------------- CREATE / LOAD FIXED SUBSET --------------------
    print("\n" + "="*90)
    print(f"📦 PREPARING FIXED {SUBSET_SIZE}‑SAMPLE SUBSET")
    print("="*90)

    test_dataset = test_dl.dataset
    total_samples = len(test_dataset)
    print(f"Total samples in test set: {total_samples}")

    indices = list(range(min(SUBSET_SIZE, total_samples)))
    subset_dataset = torch.utils.data.Subset(test_dataset, indices)
    subset_dl = torch.utils.data.DataLoader(
        subset_dataset,
        batch_size=test_dl.batch_size,
        shuffle=False,
        num_workers=0,           
        pin_memory=False
    )
    print(f"✅ Subset DataLoader ready – {len(subset_dataset)} samples (batch size = {subset_dl.batch_size})")

    # ---------- Load PyTorch model ----------
    print("\n" + "="*90)
    print("🤖 LOADING PYTORCH MODEL")
    print("="*90)

    pt_model = DeepSNNClassification_g11(num_classes=NUM_CLASSES).to(device)
    if os.path.exists(ckpt_path):
        pt_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        pt_model.eval()
        print(f"✅ Model loaded from: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint not found – using untrained model.")

    # ---------- 1. CREATE FLOAT HW WRAPPER ----------
    hw_float = CustomQuantizedG11(pt_model, quantization=False)
    print("✅ Hardware wrapper (quantization OFF) created.")

    # ---------- 2. (Optional) DEEP COMPARISON – uncomment if needed ----------
    # passed, stats, pt_metrics, hw_metrics = deep_compare_subset(
    #     pt_model, hw_float, subset_dl,
    #     num_samples=len(subset_dataset),
    #     spike_mismatch_threshold=0.02,
    #     rtol=1e-3, atol=1e-3
    # )
    # print(f"\n📊 Deep comparison result: {'✅ PASS' if passed else '❌ FAIL'}")

    # ---------- 3. FULL EVALUATION – FLOAT MODEL ON SUBSET ----------
    print("\n" + "="*90)
    print("📈 FULL EVALUATION – PyTorch (float) on subset")
    print("="*90)

    float_acc, float_f1 = eval_float_model(pt_model, subset_dl, device, CLASS_NAMES)
    print(f"✅ Float baseline: Acc = {float_acc:.4f}, F1 = {float_f1:.4f}")

    # ---------- 4. FULL EVALUATION – HW (QUANT OFF) ON SUBSET (optional) ----------
    # start = time.time()
    # hw_float_acc, hw_float_f1 = eval_hw_model(hw_float, subset_dl, CLASS_NAMES, verbose=False)
    # elapsed = time.time() - start
    # print(f"⏱️  HW (Quant OFF) total time: {elapsed:.2f}s")
    # print("="*90)

    # ---------- 5. FULL EVALUATION – HW (QUANT ON) LOOP OVER BIT WIDTHS ----------
    print("\n" + "="*90)
    print("📈 FULL EVALUATION – HW wrapper (quantization ON) with varying bit widths")
    print("="*90)



    int_values = [8,9,10]                # integer bits to try
    frac_values = [2,3,4,5, 6, 7, 8, 9, 10]  # fractional bits

    int_list = []
    frac_bits_list = []

    for int_bits in int_values:
        for frac_bits in frac_values:
           int_values.append(int_bits)
           frac_bits_list.append(frac_bits)
    total_bits_list = [i + f for i, f in zip(int_list, frac_bits_list)]


    # Store results for summary
    results = []

    for total_bits, frac_bits in zip(total_bits_list, frac_bits_list):
        print(f"\n--- Quantization: total_bits = {total_bits}, frac_bits = {frac_bits} ---")
        hw_quant = CustomQuantizedG11(
            pt_model,
            quantization=True,
            total_bits=total_bits,
            frac_bits=frac_bits
        )

        start = time.time()
        acc, f1 = eval_hw_model(hw_quant, subset_dl, CLASS_NAMES, verbose=False)
        elapsed = time.time() - start
        print(f"⏱️  HW (Quant ON, total={total_bits}, frac={frac_bits}) total time: {elapsed:.2f}s")
        print("-" * 90)

        results.append({
            'total_bits': total_bits,
            'frac_bits': frac_bits,
            'accuracy': acc,
            'f1': f1,
            'time': elapsed
        })

    # ---------- SUMMARY TABLE AND RECOMMENDATION ----------
    print("\n" + "="*90)
    print("📊 QUANTIZATION TRACKING SUMMARY")
    print("="*90)
    print(f"{'Bits':<6} {'Frac':<6} {'Accuracy':<10} {'F1':<8} {'Time (s)':<8} {'vs Float':<10}")
    print("-" * 90)

    best_bits = None
    best_acc = 0.0
    for r in results:
        acc_ratio = r['accuracy'] / float_acc if float_acc > 0 else 0
        print(f"{r['total_bits']:<6} {r['frac_bits']:<6} {r['accuracy']:<10.4f} {r['f1']:<8.4f} {r['time']:<8.2f} {acc_ratio*100:>6.1f}%")
        if r['accuracy'] > best_acc:
            best_acc = r['accuracy']
            best_bits = r['total_bits']

    print("="*90)

    # Find the smallest bit‑width that maintains at least ACC_TOLERANCE * float_acc
    recommended = None
    for r in sorted(results, key=lambda x: x['total_bits']):
        if r['accuracy'] >= ACC_TOLERANCE * float_acc:
            recommended = r
            break

    if recommended:
        print(f"\n✅ Recommendation: total_bits = {recommended['total_bits']} "
              f"(frac_bits = {recommended['frac_bits']}) – "
              f"accuracy = {recommended['accuracy']:.4f} "
              f"(≥ {ACC_TOLERANCE*100:.0f}% of float baseline {float_acc:.4f})")
    else:
        print(f"\n⚠️  No configuration meets the tolerance of {ACC_TOLERANCE*100:.0f}% of float accuracy.")
        if results:
            best = max(results, key=lambda x: x['accuracy'])
            print(f"   Best achieved: total_bits = {best['total_bits']} "
                  f"with accuracy {best['accuracy']:.4f} "
                  f"({best['accuracy']/float_acc*100:.1f}% of float)")

    # ---------- SAVE RESULTS TO CSV FOR PLOTTING ----------
    csv_file = "quantization_results.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['total_bits', 'frac_bits', 'accuracy', 'f1', 'time_seconds', 'acc_ratio'])
        # Float baseline
        writer.writerow(['float', 'float', float_acc, float_f1, 0.0, 1.0])
        for r in results:
            writer.writerow([
                r['total_bits'],
                r['frac_bits'],
                f"{r['accuracy']:.6f}",
                f"{r['f1']:.6f}",
                f"{r['time']:.3f}",
                f"{r['accuracy']/float_acc:.6f}"
            ])
    print(f"\n✅ Results saved to {csv_file}")

    print("\n" + "="*90)
    print("✅ ALL EVALUATIONS COMPLETE")
    print("="*90)



import os
import time
import torch


def final_big_test():

    # -------------------- CONFIGURE THESE --------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/student_epoch_80_f1_0.7315_acc_0.7574_g11.pt"

    NUM_CLASSES = 4
    CLASS_NAMES = [
        'negative_samples',
        'drifting_or_skidding',
        'other_crash',
        'collision'
    ]

    SUBSET_SIZE = 500

    # -------------------- LOAD TEST DATALOADER --------------------
    from kd import fourth_step
    _, test_dl = fourth_step(batch_size=1)

    # -------------------- HEADER --------------------
    print("\n" + "=" * 90)
    print(f"📦 PREPARING FIXED {SUBSET_SIZE}-SAMPLE SUBSET")
    print("=" * 90)

    # -------------------- LOAD MODEL --------------------
    print("\n" + "=" * 90)
    print("🤖 LOADING PYTORCH MODEL")
    print("=" * 90)

    pt_model = DeepSNNClassification_g11(num_classes=NUM_CLASSES).to(device)

    if os.path.exists(ckpt_path):
        pt_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        pt_model.eval()
        print(f"✅ Model loaded from: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint not found – using untrained model.")

    # -------------------- FLOAT BASELINE --------------------
    print("\n" + "=" * 90)
    print("📈 FULL EVALUATION – PyTorch (float) on subset")
    print("=" * 90)

    float_acc, float_f1 = eval_float_model(
        pt_model,
        test_dl,
        device,
        CLASS_NAMES,
        end_batch=SUBSET_SIZE
    )

    print(f"\n✅ Float baseline: Acc = {float_acc:.4f}, F1 = {float_f1:.4f}")

    # -------------------- SAFETY: RESET CUDA + DATALOADER --------------------
    torch.cuda.empty_cache()
    if device == "cuda":
        torch.cuda.synchronize()

    # Reload dataloader to avoid exhaustion issues
    from kd import fourth_step
    _, test_dl = fourth_step(batch_size=1)

    # -------------------- QUANTIZATION TESTS --------------------
    print("\n" + "=" * 90)
    print("📈 FULL EVALUATION – HW wrapper (quantization ON)")
    print("=" * 90)

    int_list = [9, 10]
    frac_bits_list = [10, 10]
    total_bits_list = [i + f for i, f in zip(int_list, frac_bits_list)]

    results = []

    for total_bits, frac_bits in zip(total_bits_list, frac_bits_list):

        print("\n" + "-" * 90)
        print(f"🚀 Quantization: total_bits = {total_bits}, frac_bits = {frac_bits}")
        print("-" * 90)

        print("Creating quantized wrapper...")
        hw_quant = CustomQuantizedG11(
            pt_model,
            quantization=True,
            total_bits=total_bits,
            frac_bits=frac_bits
        )

        print("Starting HW evaluation...")
        start = time.time()

        acc, f1 = eval_hw_model(
            hw_quant,
            test_dl,
            CLASS_NAMES,
            verbose=False,
            end_batch=SUBSET_SIZE
        )

        elapsed = time.time() - start

        print(f"⏱️  Total time: {elapsed:.2f}s")
        print(f"📊 Acc = {acc:.4f}, F1 = {f1:.4f}")

        results.append({
            "total_bits": total_bits,
            "frac_bits": frac_bits,
            "accuracy": acc,
            "f1": f1,
            "time": elapsed
        })

        # Reset dataloader for next iteration
        _, test_dl = fourth_step(batch_size=1)

    # -------------------- SUMMARY --------------------
    print("\n" + "=" * 90)
    print("📊 QUANTIZATION TRACKING SUMMARY")
    print("=" * 90)

    print(f"{'Bits':<6} {'Frac':<6} {'Accuracy':<10} {'F1':<8} {'Time(s)':<8} {'vs Float':<10}")
    print("-" * 90)

    best_bits = None
    best_acc = 0.0

    for r in results:
        acc_ratio = (r["accuracy"] / float_acc) * 100 if float_acc > 0 else 0

        print(
            f"{r['total_bits']:<6} "
            f"{r['frac_bits']:<6} "
            f"{r['accuracy']:<10.4f} "
            f"{r['f1']:<8.4f} "
            f"{r['time']:<8.2f} "
            f"{acc_ratio:>6.1f}%"
        )

        if r["accuracy"] > best_acc:
            best_acc = r["accuracy"]
            best_bits = r["total_bits"]

    print("=" * 90)

    print(f"\n🏆 Best quantization setting: {best_bits} bits "
          f"(Acc = {best_acc:.4f})")

    print("\n✨ Evaluation complete.\n")


    

   

if __name__ == "__main__":
    final_big_test()
