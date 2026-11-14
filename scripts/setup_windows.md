# Running FoundationPose natively on Windows

No Docker, WSL, conda or system CUDA install, and no admin rights. Tested on Windows 11 with an
RTX 3070 (8 GB), Python 3.10 and Visual Studio 2026 with the C++ tools.

You need Python 3.10, Visual Studio with the "Desktop development with C++" workload, and Git.

## 1. Python environment

```bash
py -3.10 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install --upgrade pip wheel "setuptools<81"
```

```bash
.venv/Scripts/python.exe -m pip install "torch==2.9.1+cu130" "torchvision==0.24.1+cu130" --index-url https://download.pytorch.org/whl/cu130
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements-lock.txt
```

## 2. CUDA compiler, from pip

These wheels install `nvcc` and the CUDA headers inside the venv. They are only needed to compile nvdiffrast.

```bash
.venv/Scripts/python.exe -m pip install nvidia-cuda-nvcc nvidia-cuda-crt nvidia-nvvm nvidia-cuda-runtime nvidia-cuda-cccl nvidia-cublas nvidia-cusparse nvidia-cusolver nvidia-curand nvidia-cufft nvidia-nvjitlink nvidia-cuda-nvrtc
```

## 3. Build nvdiffrast, get the BOP toolkit

```bash
git clone --branch v0.4.0 https://github.com/NVlabs/nvdiffrast third_party/nvdiffrast
```

```bash
cmd //c "scripts\\build_nvdiffrast_windows.bat"
```

```bash
git clone https://github.com/thodan/bop_toolkit third_party/bop_toolkit
```

```bash
git -C third_party/bop_toolkit checkout cea62d6
```

`build_nvdiffrast_windows.bat` assumes Visual Studio 2026 Community and an RTX 30-series GPU. Edit
the `vcvars64.bat` path and `TORCH_CUDA_ARCH_LIST` if yours differ.

## 4. Data

| What | Where it goes | Source |
|---|---|---|
| Weights | `weights/2023-10-28-18-33-37`, `weights/2024-01-11-20-02-45` | Google Drive link in `readme.md` |
| Demo | `demo_data/mustard0` | Google Drive link in `readme.md` |
| YCB-V | `data/BOP/ycbv/` with `test/`, `ycbv_models/`, `test_targets_bop19.json` | `bop-benchmark/ycbv` on Hugging Face: `ycbv_base`, `ycbv_models`, `ycbv_test_bop19` |
| LM-O | `data/BOP/lmo/` with `lmo_test_bop19/test/`, `models/`, `models_eval/`, `test_targets_bop19.json` | `bop-benchmark/lmo` on Hugging Face: `lmo_base`, `lmo_models`, `lmo_test_bop19` |
| CNOS masks | `data/BOP/detections/` | `bop-benchmark/bop_extra` on Hugging Face: `default_detections/classic_bop23_model_based_unseen/cnos-fastsam/` |

Keep the folder name `BOP` in capitals: the dataset readers look for it in the path.

## 5. Check it works

```bash
.venv/Scripts/python.exe run_demo.py --headless --debug 2 --debug_dir results/demo_mustard
```

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

## What differs from the Linux setup

- **pytorch3d and mycpp are not built.** Inference only used three functions from them, and
  `compat_windows.py` replaces those. `tests/test_compat_windows.py` compares them with the real
  code when `P3D_REF` and `MYCPP_REF` point at a pytorch3d checkout and a compiled mycpp. Without
  those two variables, those comparisons are skipped.
- **opencv is held below version 5**, which rejects float images in `cv2.putText`.
- **`run_demo.py --headless`** skips the preview window.

## Things that will bite you

- **Run one GPU job at a time on an 8 GB card.** FoundationPose peaks around 6.6 GB and already
  keeps the GPU near 100%. Two copies at once ran about 20 times slower than one, because Windows
  starts paging GPU memory.
- Warp prints a deprecation notice for Python 3.10. It is harmless at Warp 1.19, which is pinned.
