# ─────────────────────────────────────────────────────────────────────────────
# Forex Scaling Model — GPU training image (generic Docker)
# Base: official PyTorch + CUDA runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime

# Prevent interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA optimisation flags
ENV CUDA_LAUNCH_BLOCKING=0
ENV TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;8.9;9.0"

WORKDIR /app

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    vim \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
    # Core scientific stack
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    # High-performance data
    polars>=0.19.0 \
    pyarrow>=14.0.0 \
    # Deep learning extras (torch pre-installed in base image)
    torchvision \
    torchaudio \
    # RL
    stable-baselines3>=2.2.0 \
    gymnasium>=0.29.0 \
    # Experiment tracking
    wandb>=0.16.0 \
    tensorboard>=2.15.0 \
    # Hyperparameter optimisation
    optuna>=3.5.0 \
    # Utilities
    loguru>=0.7.0 \
    tqdm>=4.66.0 \
    rich>=13.0.0 \
    pyyaml>=6.0 \
    python-dotenv>=1.0.0 \
    numba>=0.58.0 \
    # Storage / serialisation
    joblib>=1.3.0 \
    h5py>=3.10.0

# ── Copy project ──────────────────────────────────────────────────────────────
COPY . /app/forex_scaling_model/
WORKDIR /app/forex_scaling_model

# ── Create directories for data, checkpoints, logs ───────────────────────────
RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/checkpoints \
    /app/logs \
    /app/exports

# Symlink so the model can find data at the expected relative path
RUN ln -sf /app/data /app/forex_scaling_model/data/raw_data && \
    ln -sf /app/checkpoints /app/forex_scaling_model/checkpoints && \
    ln -sf /app/logs /app/forex_scaling_model/logs

# ── Verify GPU is accessible at build time ────────────────────────────────────
RUN python -c "import torch; print(f'PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()}')"

# Default: interactive shell — run training explicitly, e.g.:
#   docker run --gpus all -it <image> python training/train_gpu.py --model haelt --epochs 10
CMD ["bash"]
