# Use NVIDIA base for GPU support (or ubuntu:22.04 for CPU-only)
FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install system deps
RUN apt-get update && apt-get install -y \
    wget git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python and create virtual env
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH="/opt/conda/bin:$PATH"

# Create and activate env
RUN conda create -n indextts python=3.10 -y
SHELL ["conda", "run", "-n", "indextts", "/bin/bash", "-c"]

# Install PyTorch (CUDA; use --index-url https://download.pytorch.org/whl/cpu for CPU)
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other deps
RUN conda install -c conda-forge pynini==2.1.6 -y \
    && pip install WeTextProcessing --no-deps

# Copy repo and install
WORKDIR /app
COPY . .
RUN pip install -e . \
    && pip install -e ".[webui]" --no-build-isolation

# Download models at build time (faster inference; ~2-3GB)
RUN mkdir -p checkpoints \
    && huggingface-cli download IndexTeam/IndexTTS-1.5 \
      config.yaml bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
      --local-dir checkpoints

# Expose port
EXPOSE 7860

# Run web UI
CMD ["python", "webui.py", "--model_dir", "checkpoints"]
