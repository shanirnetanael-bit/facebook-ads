#!/usr/bin/env bash
# One-time setup of the video engine's tools.
set -e
pip install --break-system-packages opencv-python-headless numpy scipy librosa onnx onnxruntime "rembg[cpu]" pillow
python3 -c "from PIL import features; assert features.check('raqm'), 'Pillow needs libraqm for Hebrew'"
mkdir -p /opt/vtools/models ~/.u2net

# Speech-to-text, option A (preferred for Hebrew): whisper.cpp + ivrit.ai model (needs huggingface.co):
[ -d /opt/vtools/whisper.cpp ] || git clone --depth 1 https://github.com/ggml-org/whisper.cpp /opt/vtools/whisper.cpp
cmake -S /opt/vtools/whisper.cpp -B /opt/vtools/whisper.cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build /opt/vtools/whisper.cpp/build -j"$(nproc)" --target whisper-cli
#   curl -L -o /opt/vtools/models/ggml-ivrit-large-v3-turbo.bin <ivrit.ai ggml model URL on huggingface.co>

# Speech-to-text, option B (fallback, GitHub only): Whisper large-v3-turbo ONNX (sherpa-onnx export), used by
# engine/asr.py with cross-attention DTW word timestamps.
if [ ! -d /opt/vtools/models/sherpa-onnx-whisper-turbo ]; then
  curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-turbo.tar.bz2 \
    | tar xj -C /opt/vtools/models
fi

# Speech denoiser for the voice clean-up chain (RNNoise model for ffmpeg arnndn):
[ -f /opt/vtools/models/rnnoise-sh.rnnn ] || curl -L -o /opt/vtools/models/rnnoise-sh.rnnn \
  https://raw.githubusercontent.com/GregorR/rnnoise-models/master/somnolent-hogwash-2018-09-01/sh.rnnn

# Person-segmentation model for rembg:
[ -f ~/.u2net/u2net_human_seg.onnx ] || curl -L -o ~/.u2net/u2net_human_seg.onnx \
  https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx
