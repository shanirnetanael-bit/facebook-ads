#!/usr/bin/env bash
# One-time setup of the video engine's tools.
set -e
pip install --break-system-packages opencv-python-headless numpy "rembg[cpu]" onnxruntime pillow
python3 -c "from PIL import features; assert features.check('raqm'), 'Pillow needs libraqm for Hebrew'"
# whisper.cpp (local, free speech-to-text)
mkdir -p /opt/vtools/models
[ -d /opt/vtools/whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp /opt/vtools/whisper.cpp
cmake -S /opt/vtools/whisper.cpp -B /opt/vtools/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/vtools/whisper.cpp/build -j"$(nproc)" --target whisper-cli
# Hebrew model from ivrit.ai (needs huggingface.co reachable; confirm the exact file name on the model page):
#   curl -L -o /opt/vtools/models/ggml-ivrit-large-v3-turbo.bin \
#     https://huggingface.co/ivrit-ai/whisper-large-v3-turbo-ggml/resolve/main/ggml-model.bin
# Person-segmentation model for rembg:
mkdir -p ~/.u2net
[ -f ~/.u2net/u2net_human_seg.onnx ] || curl -L -o ~/.u2net/u2net_human_seg.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx
