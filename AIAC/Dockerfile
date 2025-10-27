FROM rocm/vllm-dev:nightly
# Install build dependencies and tmux
RUN apt-get update && apt-get install -y build-essential cmake git tmux wget && \
    rm -rf /var/lib/apt/lists/*

# Uninstall old unsloth packages and install latest
RUN pip uninstall unsloth_zoo unsloth -y || true

RUN pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth_zoo && \
    pip install --upgrade --no-cache-dir --no-deps git+https://github.com/unslothai/unsloth

# Install synthetic-data-kit (ignore distutils conflicts)
RUN pip install synthetic-data-kit --ignore-installed blinker

# Install required packages with version constraints
RUN pip install "wheel>=0.42.0" "packaging" "torchvision" "numpy" "tqdm" "psutil" "jupyter"\
    "tyro" "protobuf" "sentencepiece>=0.2.0" \
    "datasets>=3.4.1,!=4.0.*,!=4.1.0" \
    "accelerate>=0.34.1" \
    "peft>=0.7.1,!=0.11.0" \
    "huggingface_hub>=0.34.0" \
    "hf_transfer" \
    "diffusers" \
    "transformers>=4.51.3,!=4.52.0,!=4.52.1,!=4.52.2,!=4.52.3,!=4.53.0,!=4.54.0,!=4.55.0,!=4.55.1,<=4.56.2" \
    "trl>=0.7.9,!=0.9.0,!=0.9.1,!=0.9.2,!=0.9.3,!=0.15.0,!=0.19.0,<=0.23.0"

# Build and install bitsandbytes from source with ROCm support
RUN git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git /tmp/bitsandbytes && \
    cd /tmp/bitsandbytes && \
    cmake -DCOMPUTE_BACKEND=hip -DBNB_ROCM_ARCH="gfx942" -S . && \
    make && \
    pip install . && \
    cd / && \
    rm -rf /tmp/bitsandbytes

# Set environment variables for vLLM
ENV GLOO_SOCKET_IFNAME=lo
ENV NCCL_SOCKET_IFNAME=lo

# Copy the synthetic data AI agents challenge directory, eval directory is ignored in .dockerignore
COPY . /workspace/AIAC/

# Set working directory
WORKDIR /workspace

# Default command
# CMD ["/bin/bash"]
CMD ["jupyter-lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
