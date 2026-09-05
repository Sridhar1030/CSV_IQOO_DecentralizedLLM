# Running and compiling on the Qualcomm Hexagon NPU

Findings from bring-up on an attached iQOO I2501. Everything marked **VERIFIED** was
reproduced by running it on this device; everything marked **UNVERIFIED** is inference
from source or docs and has not been executed.

---

## 1. Status

| Capability | State |
|---|---|
| Model executing on Hexagon HTP, plain `adb shell`, unprivileged | **VERIFIED** |
| CPU (XNNPACK) inference on device | **VERIFIED** |
| GPU (OpenCL) inference on device | **VERIFIED** |
| On-device JIT compile to a QNN context binary | **VERIFIED** |
| Our KV-cache attention shard on NPU | **FAILS** — see §7 |
| Host-side AOT compile from macOS | **IMPOSSIBLE** — see §8 |

A 1 MB model runs fully delegated to the HTP at **3.38 ms** (297 runs), reproduced at
3.49 ms. Device proof from logcat:

```
remote_handle64_open: opened handle ... for file:///libQnnHtpV81Skel.so
    ?qnn_2_49_0_skel_handle_invoke&_modver=1.0&_dom=cdsp on domain 100000
dspqueue_create: created Queue 0, ... DSP 0x00000000 for domain 100000
QnnContext_createFromBinary done successfully
QnnMem_register done. status 0x0
```

894 `QnnGraph_execute` lines in the run log. `getenforce` = **Enforcing**, uid **2000**.

---

## 2. Device

| | |
|---|---|
| Model | iQOO I2501 (`10BFAT1U06000Z9`) |
| SoC | SM8850 — Snapdragon 8 Elite Gen 5 |
| Hexagon | **V81** |
| ABI | arm64-v8a (only) |
| Android | 16 (SDK 36), SELinux Enforcing |
| RAM | 15.5 GB |
| Vendor QAIRT | 2.34.3 in `/vendor/lib64/hw/` |

CPU features present and used: `asimddp`, `i8mm`, `fphp`, `asimdhp`. Also exposes
`sve2`/`svei8mm`, which nothing in this stack uses.

---

## 3. The working recipe

The pairing matters more than anything else here. **LiteRT nightly + QAIRT 2.49.0.**
Other combinations fail on version gates (§4).

### Binaries — Google GCS, public, no auth

```
https://storage.googleapis.com/litert/binaries/nightly/android_arm64/benchmark_model
https://storage.googleapis.com/litert/binaries/nightly/android_arm64/libLiteRtDispatch_Qualcomm.so
https://storage.googleapis.com/litert/binaries/nightly/android_arm64/libLiteRtCompilerPlugin_Qualcomm.so
```

Swap `nightly` → `latest` for a pinned channel, but **`latest` SIGBUSes on this device**;
nightly is what fixed it. Other useful artifacts in the same bucket:

| File | Purpose |
|---|---|
| `run_model` (18 MB) | Returns output **tensors**, not just latency. Use for numerical checks. |
| `npu_numerics_check` (18 MB) | Compares NPU vs CPU output. Takes `--dispatch_library_dir=`. |
| `gpu_numerics_check` | Same for GPU. |

`benchmark_model` is self-contained — its only `DT_NEEDED` entries are Android system
libs (`libandroid`, `libEGL`, `libGLESv2/v3`, `libdl`, `libm`, `liblog`, `libc`).
LiteRT is statically linked in. For CPU you push **one file**.

### QNN libraries — QAIRT 2.49.0

`libLiteRtCompilerPlugin_Qualcomm.so` hard-links `libQnnHtp.so`, `libQnnIr.so`,
`libQnnSaver.so`, `libQnnSystem.so` — it will not load without all four.

Two sources, both without a Qualcomm account:

**Maven Central** (easiest, but incomplete):

```
com.qualcomm.qti:qnn-runtime:2.49.0
```

The AAR carries 19 `.so` under `jni/arm64-v8a/` — but **not** `libQnnIr.so` or
`libQnnSaver.so`.

**QAIRT zip** (complete). The archive is 2.35 GB, but you do not need to download it.
A `HEAD` returns 403; a `GET` returns 200 and the server honours range requests, so you
can read the ZIP central directory (11,631 entries) and extract just the two files in
~30 seconds. Paths inside the zip:

```
qairt/<ver>/lib/aarch64-android/libQnnIr.so
qairt/<ver>/lib/aarch64-android/libQnnSaver.so
```
