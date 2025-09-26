FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 

RUN apt-get update && apt-get install -y git git-lfs python3 python3-pip curl && git lfs install && rm -rf /var/lib/apt/lists/* 

RUN pip3 install -U uv WORKDIR /app COPY . . 

RUN git lfs pull RUN uv sync --all-extras --default-index-url https://pypi.org/simple/ 

RUN uv tool install "huggingface-hub[cli]" && huggingface-cli download IndexTeam/IndexTTS-2 --local-dir checkpoints && chmod -R 755 checkpoints EXPOSE 7860 CMD ["uv", "run", "webui.py", "--fp16"]