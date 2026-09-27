"""
Edge-cost profiling for FRESCA-Net (reviewer #2, point 1).

Reports, for FRESCA-Net and the Piczak baseline:

  * trainable parameters and model size (fp32 / fp16 / int8)
  * multiply-accumulate operations (MACs) of the backbone, counted analytically
    with forward hooks, cross-checked against thop
  * MACs of the parameter-free log-mel front-end (STFT + mel filterbank),
    which a parameter count cannot see but an edge device still pays
  * end-to-end cost of one clip under the paper's 3-crop TTA protocol
  * measured wall-clock latency, batch size 1: CPU single-thread, CPU
    multi-thread, CUDA, and ONNX Runtime (fp32 + dynamic int8)
  * peak activation memory of a batch-1 forward pass

All latency numbers are median over N timed runs after warm-up, which is the
robust statistic for a laptop CPU with thermal/frequency drift.

Usage:
    python benchmark/profile_edge.py                     # full run
    python benchmark/profile_edge.py --runs 100          # more timing samples
    python benchmark/profile_edge.py --no-onnx           # skip ONNX section
"""
import os
import sys
import json
import time
import argparse
import statistics
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import MODELS, count_params
from frontend import LogMelFrontend, SR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

CROP_SEC = 3.0            # training / inference crop, as in the paper
CLIP_SEC = 5.0            # full ESC-50 clip
N_FFT, HOP, N_MELS = 1024, 512, 128
TTA_CROPS = 3             # paper's label-free test-time augmentation


# ---------------------------------------------------------------------------
# 1. Analytic MAC counting via forward hooks
# ---------------------------------------------------------------------------
def _conv_macs(module, inp, out):
    """MACs of a conv = output_elements * (Cin/groups * prod(kernel))."""
    out_elems = out.numel() / out.shape[0]          # per sample
    cin_per_group = module.in_channels / module.groups
    k = 1
    for d in module.kernel_size:
        k *= d
    return out_elems * cin_per_group * k


def _linear_macs(module, inp, out):
    return module.in_features * module.out_features


def count_macs(model, x):
    """
    Count MACs of one forward pass of `model` on input `x` (batch of 1).

    Only Conv1d/Conv2d/Linear are counted: they carry >99% of the arithmetic.
    BatchNorm, ReLU, SE-sigmoid, softmax and pooling are elementwise/reduction
    ops whose cost is reported separately below so nothing is silently dropped.
    """
    totals = {"conv": 0.0, "linear": 0.0}
    per_layer = []
    handles = []

    def make_hook(name, kind, fn):
        def hook(module, inp, out):
            m = fn(module, inp, out)
            totals[kind] += m
            per_layer.append((name, kind, m))
        return hook

    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            handles.append(m.register_forward_hook(make_hook(name, "conv", _conv_macs)))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(make_hook(name, "linear", _linear_macs)))

    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()

    return totals["conv"] + totals["linear"], per_layer


def count_elementwise(model, x):
    """Elementwise/reduction op count (BN, ReLU, SE, pooling) for completeness."""
    total = {"n": 0.0}
    handles = []

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            total["n"] += out.numel() / out.shape[0]

    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.ReLU,
                          nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Sigmoid, nn.Tanh)):
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return total["n"]


# ---------------------------------------------------------------------------
# 2. Front-end MACs (parameter-free, but real arithmetic on device)
# ---------------------------------------------------------------------------
def frontend_macs(seconds, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """
    Log-mel front-end cost for one waveform of `seconds`.

      * STFT: n_frames real FFTs of length n_fft. A radix-2 complex FFT costs
        (n_fft/2)*log2(n_fft) complex butterflies ~= 2*n_fft*log2(n_fft) real
        MACs; we use the standard 2*N*log2(N) real-MAC figure per frame, plus
        the n_fft-point window multiply.
      * power spectrum: 2 mults + 1 add per bin  -> counted as 2 MACs/bin.
      * mel projection: a dense [n_mels x n_bins] @ [n_bins x n_frames] matmul.
    """
    n_samples = int(seconds * sr)
    n_frames = n_samples // hop + 1                  # center=True padding
    n_bins = n_fft // 2 + 1

    fft = n_frames * 2.0 * n_fft * np.log2(n_fft)
    window = n_frames * n_fft
    power = n_frames * n_bins * 2.0
    mel = n_mels * n_bins * n_frames                 # dense filterbank matmul
    log_std = n_mels * n_frames * 3.0                # log + per-instance standardise

    return {
        "n_frames": int(n_frames),
        "fft": fft, "window": window, "power": power,
        "mel_matmul": mel, "log_and_standardise": log_std,
        "total": fft + window + power + mel + log_std,
    }


def mel_macs_sparse(seconds, sr=SR, n_fft=N_FFT, hop=HOP, n_mels=N_MELS):
    """Mel matmul cost if the filterbank's zeros are skipped (sparse kernel)."""
    import librosa
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=20, fmax=22050)
    nnz = int((fb > 0).sum())
    n_frames = int(seconds * sr) // hop + 1
    return nnz * n_frames, nnz


# ---------------------------------------------------------------------------
# 3. Latency measurement
# ---------------------------------------------------------------------------
def time_module(fn, runs, warmup, sync_cuda=False):
    for _ in range(warmup):
        fn()
    if sync_cuda:
        torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_cuda:
            torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)   # ms
    return {
        "median_ms": statistics.median(samples),
        "mean_ms": statistics.mean(samples),
        "p90_ms": sorted(samples)[int(0.9 * len(samples)) - 1],
        "min_ms": min(samples),
        "runs": runs,
    }


def model_size_mb(model):
    n = sum(p.numel() for p in model.parameters())
    b = sum(bf.numel() for bf in model.buffers())
    return {
        "fp32_mb": (n * 4 + b * 4) / 1024 ** 2,
        "fp16_mb": (n * 2 + b * 2) / 1024 ** 2,
        "int8_mb": (n * 1 + b * 4) / 1024 ** 2,   # weights int8, buffers fp32
    }


def peak_activation_mb(model, x):
    """Peak activation footprint of a batch-1 inference, measured by hooks."""
    live = {"cur": 0.0, "peak": 0.0}
    handles = []

    def hook(module, inp, out):
        if isinstance(out, torch.Tensor):
            mb = out.numel() * out.element_size() / 1024 ** 2
            live["cur"] += mb
            live["peak"] = max(live["peak"], live["cur"])

    for m in model.modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return live["peak"]


# ---------------------------------------------------------------------------
# 4. ONNX Runtime (fp32 + dynamic int8) — the realistic edge deployment path
# ---------------------------------------------------------------------------
def calibration_inputs(in_ch, n=32, seed=0):
    """
    Real log-mel inputs for static int8 calibration, taken from the cached
    ESC-50 waveforms so the activation ranges are the ones the deployed model
    actually sees. Falls back to noise if the cache is absent.
    """
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
    wav_path = os.path.join(cache, "waveforms.npy")
    fe = LogMelFrontend(coord=(in_ch == 2)).eval()
    crop = int(CROP_SEC * SR)
    outs = []
    if os.path.isfile(wav_path):
        W = np.load(wav_path, mmap_mode="r")
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(W), size=min(n, len(W)), replace=False)
        with torch.no_grad():
            for i in idx:
                w = torch.from_numpy(np.ascontiguousarray(W[i])).float().unsqueeze(0)
                s = rng.integers(0, max(1, w.shape[1] - crop))
                outs.append(fe(w[:, s:s + crop]).numpy())
    else:
        with torch.no_grad():
            for _ in range(n):
                outs.append(fe(torch.randn(1, crop)).numpy())
    return outs


def onnx_bench(model, x, tag, runs, warmup, outdir):
    try:
        import onnxruntime as ort
        from onnxruntime.quantization import (quantize_static, QuantType,
                                              QuantFormat, CalibrationDataReader)
    except Exception as e:                                    # pragma: no cover
        return {"error": f"onnxruntime unavailable: {e}"}

    os.makedirs(outdir, exist_ok=True)
    fp32_path = os.path.join(outdir, f"{tag}_fp32.onnx")
    int8_path = os.path.join(outdir, f"{tag}_int8.onnx")

    model.eval()
    # The legacy TorchScript exporter cannot lower adaptive_avg_pool2d when the
    # output size is not a factor of the input (hits the Piczak baseline's
    # AdaptiveAvgPool2d((4,4))); fall back to the dynamo exporter in that case.
    try:
        torch.onnx.export(model, (x,), fp32_path, input_names=["x"],
                          output_names=["logits"], opset_version=17, dynamo=False)
    except Exception as e_ts:
        try:
            prog = torch.onnx.export(model, (x,), input_names=["x"],
                                     output_names=["logits"], dynamo=True)
            prog.save(fp32_path)
        except Exception as e_dyn:
            return {"error": f"onnx export failed (torchscript: {e_ts.__class__.__name__}; "
                             f"dynamo: {e_dyn.__class__.__name__}: {e_dyn})"}

    out, meta = {}, {}
    xn = x.numpy()

    def run_sess(path, label):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1                # single core = edge-honest
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        r = time_module(lambda: sess.run(None, {name: xn}), runs, warmup)
        r["file_mb"] = os.path.getsize(path) / 1024 ** 2
        out[label] = r

    run_sess(fp32_path, "onnx_fp32_1thread")

    # Static (calibrated) int8, NOT dynamic: ONNX Runtime has no fast int8
    # kernel for dynamically-quantized Conv, so quantize_dynamic on a conv net
    # inserts Quantize/Dequantize pairs around every conv and runs an order of
    # magnitude SLOWER than fp32. Static QDQ quantization folds the scales into
    # QLinearConv and is the representative edge-deployment path.
    in_name = "x"

    class _Reader(CalibrationDataReader):
        def __init__(self, batches):
            self.it = iter([{in_name: b} for b in batches])

        def get_next(self):
            return next(self.it, None)

    try:
        # Pre-process (shape inference + BN folding + constant folding) first.
        # Without it the conv biases are not initializers, the quantizer cannot
        # build QLinearConv, and the "int8" graph is slower than fp32.
        prep_path = os.path.join(outdir, f"{tag}_fp32_prep.onnx")
        try:
            from onnxruntime.quantization.shape_inference import quant_pre_process
            quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)
            src = prep_path
        except Exception:
            src = fp32_path

        cal = calibration_inputs(x.shape[1], n=32)
        quantize_static(src, int8_path, _Reader(cal),
                        quant_format=QuantFormat.QDQ,
                        activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8,
                        per_channel=True)
        run_sess(int8_path, "onnx_int8_static_1thread")
        meta["int8_calibration_clips"] = len(cal)
        meta["int8_preprocessed"] = (src == prep_path)
    except Exception as e:
        out["onnx_int8_static_1thread"] = {"error": f"{e.__class__.__name__}: {e}"}
    out["_meta"] = meta
    return out


# ---------------------------------------------------------------------------
def profile_model(name, runs, warmup, do_onnx, outdir):
    in_ch = 2 if name == "fresca" else 1
    kw = {"in_ch": in_ch} if name == "fresca" else {}
    model = MODELS[name](n_classes=50, **kw).eval()

    n_frames_crop = int(CROP_SEC * SR) // HOP + 1
    n_frames_clip = int(CLIP_SEC * SR) // HOP + 1
    x_crop = torch.randn(1, in_ch, N_MELS, n_frames_crop)
    x_clip = torch.randn(1, in_ch, N_MELS, n_frames_clip)

    macs_crop, per_layer = count_macs(model, x_crop)
    macs_clip, _ = count_macs(model, x_clip)
    elem_crop = count_elementwise(model, x_crop)

    # cross-check against thop
    thop_macs = None
    try:
        import thop
        m2 = MODELS[name](n_classes=50, **kw).eval()
        thop_macs, _ = thop.profile(m2, inputs=(x_crop.clone(),), verbose=False)
    except Exception as e:
        thop_macs = f"unavailable: {e}"

    fe_crop = frontend_macs(CROP_SEC)
    fe_sparse, nnz = mel_macs_sparse(CROP_SEC)

    res = {
        "model": name,
        "params": count_params(model),
        "params_millions": count_params(model) / 1e6,
        "size": model_size_mb(model),
        "input_shape_crop": list(x_crop.shape),
        "input_shape_clip": list(x_clip.shape),
        "backbone_macs_crop": macs_crop,
        "backbone_macs_crop_M": macs_crop / 1e6,
        "backbone_macs_clip_M": macs_clip / 1e6,
        "backbone_elementwise_ops_M": elem_crop / 1e6,
        "thop_macs_crop_M": (thop_macs / 1e6) if isinstance(thop_macs, float) else thop_macs,
        "frontend_macs_crop": fe_crop,
        "frontend_macs_crop_M": fe_crop["total"] / 1e6,
        "frontend_mel_sparse_macs_M": fe_sparse / 1e6,
        "frontend_mel_nnz": nnz,
        "end_to_end_macs_1crop_M": (macs_crop + fe_crop["total"]) / 1e6,
        "end_to_end_macs_3crop_M": TTA_CROPS * (macs_crop + fe_crop["total"]) / 1e6,
        "peak_activation_mb_crop": peak_activation_mb(model, x_crop),
        "latency": {},
    }

    # ---- PyTorch CPU, single thread (edge-honest) ----
    n_cpu = os.cpu_count() or 1
    torch.set_num_threads(1)
    with torch.no_grad():
        res["latency"]["torch_cpu_1thread"] = time_module(
            lambda: model(x_crop), runs, warmup)

    # ---- PyTorch CPU, all threads ----
    torch.set_num_threads(n_cpu)
    with torch.no_grad():
        res["latency"][f"torch_cpu_{n_cpu}thread"] = time_module(
            lambda: model(x_crop), runs, warmup)
    torch.set_num_threads(1)

    # ---- front-end on CPU (waveform -> logmel), single thread ----
    fe = LogMelFrontend(coord=(name == "fresca")).eval()
    wav = torch.randn(1, int(CROP_SEC * SR))
    with torch.no_grad():
        res["latency"]["frontend_cpu_1thread"] = time_module(
            lambda: fe(wav), runs, warmup)

    # ---- CUDA ----
    if torch.cuda.is_available():
        gm = MODELS[name](n_classes=50, **kw).eval().cuda()
        gx = x_crop.cuda()
        with torch.no_grad():
            res["latency"]["cuda"] = time_module(
                lambda: gm(gx), runs, warmup, sync_cuda=True)
        res["cuda_device"] = torch.cuda.get_device_name(0)
        del gm, gx
        torch.cuda.empty_cache()

    # ---- ONNX Runtime ----
    if do_onnx:
        ob = onnx_bench(model, x_crop, name, runs, warmup, outdir)
        if "error" in ob and isinstance(ob.get("error"), str):
            res["onnx_error"] = ob["error"]          # whole section unavailable
        else:
            res["onnx_meta"] = ob.pop("_meta", {})
            res["latency"].update(ob)

    # heaviest layers, for the discussion
    per_layer.sort(key=lambda t: -t[2])
    res["top_layers_macs_M"] = [{"layer": n, "macs_M": m / 1e6}
                                for n, _, m in per_layer[:8]]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--no-onnx", action="store_true")
    ap.add_argument("--models", nargs="+", default=["fresca", "piczak"])
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "edge_profile.json"))
    args = ap.parse_args()

    torch.manual_seed(0)
    outdir = os.path.join(os.path.dirname(args.out), "onnx")

    report = {
        "protocol": {
            "crop_sec": CROP_SEC, "clip_sec": CLIP_SEC, "sr": SR,
            "n_fft": N_FFT, "hop": HOP, "n_mels": N_MELS, "tta_crops": TTA_CROPS,
            "batch_size": 1,
        },
        "host": {
            "cpu_count": os.cpu_count(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "models": {},
    }
    try:
        import platform
        report["host"]["platform"] = platform.platform()
        report["host"]["processor"] = platform.processor()
        # platform.processor() only gives the family/model on Windows; the
        # paper quotes the marketing name, so read the real one.
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.run(
                ["reg", "query",
                 r"HKLM\HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                 "/v", "ProcessorNameString"],
                capture_output=True, text=True, timeout=20).stdout
            for line in out.splitlines():
                if "ProcessorNameString" in line:
                    report["host"]["cpu_name"] = line.split("REG_SZ")[-1].strip()
    except Exception:
        pass

    for name in args.models:
        print(f"\n=== profiling {name} ===", flush=True)
        r = profile_model(name, args.runs, args.warmup, not args.no_onnx, outdir)
        report["models"][name] = r
        print(f"  params        : {r['params_millions']:.3f} M")
        print(f"  backbone MACs : {r['backbone_macs_crop_M']:.1f} M  (3 s crop)")
        print(f"  thop cross-chk: {r['thop_macs_crop_M']}")
        print(f"  front-end MACs: {r['frontend_macs_crop_M']:.1f} M")
        print(f"  e2e 3-crop    : {r['end_to_end_macs_3crop_M']:.1f} M")
        if "onnx_error" in r:
            print(f"  onnx            : {r['onnx_error']}")
        for k, v in r["latency"].items():
            if isinstance(v, dict) and "error" in v:
                print(f"  {k:24s}: {v['error']}")
            else:
                print(f"  {k:24s}: {v['median_ms']:.2f} ms (median)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
