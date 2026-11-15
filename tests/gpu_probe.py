"""Thunder Compute virtualization probe (§6.8 rule 9, Gate A).

Thunder virtualizes GPUs over the network, so we check that compute and memory bandwidth are
within 25% of a bare-metal A6000. Bare-metal reference = spec x typical achieved efficiency
(bf16 dense tensor 154.8 TFLOPS x 0.70 from cuBLAS; 768 GB/s x 0.90 for device copies).
Also times host-to-device copies, which virtualization affects most.

  python tests/gpu_probe.py
"""
import sys
import time

import torch

REF_TFLOPS = 154.8 * 0.70
REF_GBPS = 768 * 0.90
PASS = 0.75


def timeit(fn, reps):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t = time.perf_counter()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / reps
    return max(wall, start.elapsed_time(end) / 1e3 / reps)


def main() -> int:
    if not torch.cuda.is_available():
        print("no CUDA device")
        return 1
    print("device:", torch.cuda.get_device_name(0))
    n = 8192
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    tflops = 2 * n ** 3 / timeit(lambda: a @ b, 20) / 1e12
    # Never-written buffers can be elided by the virtualization layer, so copy real data and check it.
    x = torch.randint(0, 255, (512 * 2 ** 20,), device="cuda", dtype=torch.uint8)
    y = torch.zeros_like(x)
    gbps = 2 * x.numel() / timeit(lambda: y.copy_(x), 20) / 1e9
    if not torch.equal(x[:: 2 ** 16], y[:: 2 ** 16]):
        print("device copy did not land")
        return 1
    h = torch.randint(0, 255, (512 * 2 ** 20,), dtype=torch.uint8).pin_memory()
    h2d = h.numel() / timeit(lambda: x.copy_(h, non_blocking=True), 10) / 1e9
    ok_c, ok_m = tflops >= PASS * REF_TFLOPS, gbps >= PASS * REF_GBPS
    print(f"bf16 GEMM   {tflops:7.1f} TFLOPS  (bare-metal ref {REF_TFLOPS:.0f}; {tflops / REF_TFLOPS:5.0%}) {'ok' if ok_c else 'LOW'}")
    print(f"device copy {gbps:7.1f} GB/s    (bare-metal ref {REF_GBPS:.0f}; {gbps / REF_GBPS:5.0%}) {'ok' if ok_m else 'LOW'}")
    print(f"host->dev   {h2d:7.1f} GB/s    (PCIe 4.0 x16 bare metal ~ 25; informational)")
    print("PASS" if ok_c and ok_m else "FAIL: move to a bare-metal provider (RunPod / Vast) per §6.8")
    return 0 if ok_c and ok_m else 1


if __name__ == "__main__":
    sys.exit(main())
