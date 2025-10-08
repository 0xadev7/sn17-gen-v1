#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# Parse Args
# -------------------------
TEMP=$(getopt -o h --long help,new-env,basic,train,xformers,flash-attn,diffoctreerast,vox2seq,spconv,mipgaussian,kaolin,nvdiffrast,demo -n 'setup.sh' -- "$@")
eval set -- "$TEMP"

HELP=false
NEW_ENV=false
BASIC=false
TRAIN=false
XFORMERS=false
FLASHATTN=false
DIFFOCTREERAST=false
VOX2SEQ=false
LINEAR_ASSIGNMENT=false
SPCONV=false
ERROR=false
MIPGAUSSIAN=false
KAOLIN=false
NVDIFFRAST=false
DEMO=false

if [ "$#" -eq 1 ]; then
  HELP=true
fi

while true; do
  case "$1" in
    -h|--help) HELP=true; shift ;;
    --new-env) NEW_ENV=true; shift ;;
    --basic) BASIC=true; shift ;;
    --train) TRAIN=true; shift ;;
    --xformers) XFORMERS=true; shift ;;
    --flash-attn) FLASHATTN=true; shift ;;
    --diffoctreerast) DIFFOCTREERAST=true; shift ;;
    --vox2seq) VOX2SEQ=true; shift ;;
    --spconv) SPCONV=true; shift ;;
    --mipgaussian) MIPGAUSSIAN=true; shift ;;
    --kaolin) KAOLIN=true; shift ;;
    --nvdiffrast) NVDIFFRAST=true; shift ;;
    --demo) DEMO=true; shift ;;
    --) shift; break ;;
    *) ERROR=true; break ;;
  esac
done

if [ "$ERROR" = true ]; then
  echo "Error: Invalid argument"
  HELP=true
fi

if [ "$HELP" = true ]; then
  cat <<EOF
Usage: setup.sh [OPTIONS]
Options:
  -h, --help              Display this help message
  --new-env               Create a new conda environment (CUDA 12.4 wheels; OK on CUDA 12.8 hosts)
  --basic                 Install basic dependencies
  --train                 Install training dependencies (pillow-simd etc.)
  --xformers              Install xformers (auto-select cu118/cu121/cu124)
  --flash-attn            Install flash-attn
  --diffoctreerast        Install diffoctreerast
  --vox2seq               Install vox2seq
  --spconv                Install spconv
  --mipgaussian           Install mip-splatting
  --kaolin                Install kaolin
  --nvdiffrast            Install nvdiffrast
  --demo                  Install all dependencies for demo
EOF
  exit 0
fi

# -------------------------
# New Env (CUDA 12.4 wheels -> works on CUDA 12.8)
# -------------------------
if [ "$NEW_ENV" = true ]; then
  conda create -y -n trellis python=3.10
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate trellis
  # Use PyTorch + CUDA 12.4 wheels (compatible with 12.8 drivers)
  conda install -y pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=12.4 -c pytorch -c nvidia
fi

# -------------------------
# System Info
# -------------------------
WORKDIR=$(pwd)
PYTORCH_VERSION=$(python -c "import torch; print(getattr(torch, '__version__', 'none'))" || echo "none")
HAS_TORCH=$(python - <<'PY'
try:
  import torch
  print(1)
except Exception:
  print(0)
PY
)

PLATFORM="cpu"
CUDA_VERSION=""
CUDA_MAJOR_VERSION=""
CUDA_MINOR_VERSION=""

if [ "$HAS_TORCH" = "1" ]; then
  PLATFORM=$(python - <<'PY'
import torch
print(('cuda' if torch.version.cuda else ('hip' if torch.version.hip else 'unknown')) if torch.cuda.is_available() else 'cpu')
PY
)
  if [ "$PLATFORM" = "cuda" ]; then
    CUDA_VERSION=$(python -c "import torch; print(torch.version.cuda)")
    CUDA_MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d'.' -f1)
    CUDA_MINOR_VERSION=$(echo "$CUDA_VERSION" | cut -d'.' -f2)
    echo "[SYSTEM] PyTorch: $PYTORCH_VERSION  |  CUDA build: $CUDA_VERSION"
  elif [ "$PLATFORM" = "hip" ]; then
    HIP_VERSION=$(python -c "import torch; print(torch.version.hip)")
    HIP_MAJOR_VERSION=$(echo "$HIP_VERSION" | cut -d'.' -f1)
    HIP_MINOR_VERSION=$(echo "$HIP_VERSION" | cut -d'.' -f2)
    if [ "$PYTORCH_VERSION" != "2.4.1+rocm6.1" ]; then
      echo "[SYSTEM] Installing PyTorch 2.4.1 for HIP ($PYTORCH_VERSION -> 2.4.1+rocm6.1)"
      pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/rocm6.1 --user
      mkdir -p /tmp/extensions
      sudo cp -r /opt/rocm/share/amd_smi /tmp/extensions/amd_smi
      cd /tmp/extensions/amd_smi && sudo chmod -R 777 . && pip install .
      cd "$WORKDIR"
      PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
    fi
    echo "[SYSTEM] PyTorch: $PYTORCH_VERSION  |  HIP: $HIP_VERSION"
  fi
fi

# Wheel CUDA tag resolver (for xformers etc.)
WHEEL_CUDA="cpu"
if [ "$PLATFORM" = "cuda" ]; then
  if [ "$CUDA_MAJOR_VERSION" = "11" ]; then
    WHEEL_CUDA="cu118"
  elif [ "$CUDA_MAJOR_VERSION" = "12" ]; then
    # Use cu124 wheels for CUDA 12.4+ (good on 12.8), else fallback to cu121
    if [ "${CUDA_MINOR_VERSION:-0}" -ge 4 ]; then
      WHEEL_CUDA="cu124"
    else
      WHEEL_CUDA="cu121"
    fi
  fi
fi
echo "[RESOLVE] Wheel CUDA tag: $WHEEL_CUDA"

# -------------------------
# Basic deps
# -------------------------
if [ "$BASIC" = true ]; then
  pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless scipy ninja rembg onnxruntime trimesh open3d xatlas pyvista pymeshfix igraph transformers
  pip install "git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8"
fi

# -------------------------
# Train deps
# -------------------------
if [ "$TRAIN" = true ]; then
  pip install tensorboard pandas lpips
  pip uninstall -y pillow || true
  sudo apt-get update && sudo apt-get install -y libjpeg-dev
  pip install pillow-simd
fi

# -------------------------
# xformers (uses WHEEL_CUDA)
# -------------------------
if [ "$XFORMERS" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    case "$WHEEL_CUDA" in
      cu118) IDX_URL="https://download.pytorch.org/whl/cu118" ;;
      cu121) IDX_URL="https://download.pytorch.org/whl/cu121" ;;
      cu124) IDX_URL="https://download.pytorch.org/whl/cu124" ;;
      *) echo "[XFORMERS] Unsupported CUDA tag: $WHEEL_CUDA"; IDX_URL="";;
    esac

    if [ -n "${IDX_URL}" ]; then
      case "$PYTORCH_VERSION" in
        2.0.1) VER="0.0.22" ;;
        2.1.0) VER="0.0.22.post7" ;;
        2.1.1) VER="0.0.23" ;;
        2.1.2) VER="0.0.23.post1" ;;
        2.2.0) VER="0.0.24" ;;
        2.2.1) VER="0.0.25" ;;
        2.2.2) VER="0.0.25.post1" ;;
        2.3.0) VER="0.0.26.post1" ;;
        2.4.0) VER="0.0.27.post2" ;;
        2.4.1) VER="0.0.28" ;;
        2.5.0) VER="0.0.28.post2" ;;
        2.5.1) VER="0.0.28.post2" ;;
        *) echo "[XFORMERS] Unknown PyTorch $PYTORCH_VERSION; trying latest known for $WHEEL_CUDA"; VER="0.0.28.post2" ;;
      esac
      pip install "xformers==${VER}" --index-url "${IDX_URL}" || {
        echo "[XFORMERS] Fallback: try plain pip (may build from source)"; pip install "xformers==${VER}";
      }
    fi
  elif [ "$PLATFORM" = "hip" ]; then
    case "$PYTORCH_VERSION" in
      2.4.1+rocm6.1) pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/rocm6.1 ;;
      *) echo "[XFORMERS] Unsupported PyTorch on HIP: $PYTORCH_VERSION" ;;
    esac
  else
    echo "[XFORMERS] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# flash-attn
# -------------------------
if [ "$FLASHATTN" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    # Prebuilt wheels support CUDA 12.x. If it tries to build, add: --no-build-isolation
    pip install flash-attn || pip install flash-attn --no-build-isolation
  elif [ "$PLATFORM" = "hip" ]; then
    echo "[FLASHATTN] Prebuilt binaries not found. Building from source..."
    mkdir -p /tmp/extensions
    git clone --recursive https://github.com/ROCm/flash-attention.git /tmp/extensions/flash-attention
    cd /tmp/extensions/flash-attention
    git checkout tags/v2.6.3-cktile
    GPU_ARCHS=gfx942 python setup.py install
    cd "$WORKDIR"
  else
    echo "[FLASHATTN] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# kaolin
# -------------------------
if [ "$KAOLIN" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    case "$PYTORCH_VERSION" in
      2.0.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html ;;
      2.1.0) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.0_cu118.html ;;
      2.1.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.1_cu118.html ;;
      2.2.0) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.0_cu118.html ;;
      2.2.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.1_cu118.html ;;
      2.2.2) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.2.2_cu118.html ;;
      2.4.0|2.4.1|2.5.0|2.5.1) pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html ;;
      *) echo "[KAOLIN] Unsupported PyTorch version for prebuilt index: $PYTORCH_VERSION" ;;
    esac
  else
    echo "[KAOLIN] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# nvdiffrast
# -------------------------
if [ "$NVDIFFRAST" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    mkdir -p /tmp/extensions
    git clone https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast
    pip install /tmp/extensions/nvdiffrast
  else
    echo "[NVDIFFRAST] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# diffoctreerast
# -------------------------
if [ "$DIFFOCTREERAST" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    mkdir -p /tmp/extensions
    git clone --recurse-submodules https://github.com/JeffreyXiang/diffoctreerast.git /tmp/extensions/diffoctreerast
    pip install /tmp/extensions/diffoctreerast
  else
    echo "[DIFFOCTREERAST] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# mip-splatting
# -------------------------
if [ "$MIPGAUSSIAN" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    mkdir -p /tmp/extensions
    git clone https://github.com/autonomousvision/mip-splatting.git /tmp/extensions/mip-splatting
    pip install /tmp/extensions/mip-splatting/submodules/diff-gaussian-rasterization/
  else
    echo "[MIPGAUSSIAN] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# vox2seq
# -------------------------
if [ "$VOX2SEQ" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    mkdir -p /tmp/extensions
    cp -r extensions/vox2seq /tmp/extensions/vox2seq
    pip install /tmp/extensions/vox2seq
  else
    echo "[VOX2SEQ] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# spconv
# -------------------------
if [ "$SPCONV" = true ]; then
  if [ "$PLATFORM" = "cuda" ]; then
    case "$CUDA_MAJOR_VERSION" in
      11) pip install spconv-cu118 ;;
      12) pip install spconv-cu120 ;;  # cu120 wheel works across CUDA 12.x, incl. 12.8
      *) echo "[SPCONV] Unsupported CUDA major version: $CUDA_MAJOR_VERSION" ;;
    esac
  else
    echo "[SPCONV] Unsupported platform: $PLATFORM"
  fi
fi

# -------------------------
# demo
# -------------------------
if [ "$DEMO" = true ]; then
  pip install gradio==4.44.1 gradio_litmodel3d==0.0.1
fi

echo "[DONE] Setup completed."
