#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="output"
mkdir -p "${OUT_DIR}"

if [[ ! -f data/train-images-idx3-ubyte.gz || ! -f data/train-labels-idx1-ubyte.gz ]]; then
  echo "MNIST data not found. Downloading..."
  bash scripts/download_mnist.sh
fi

make

./bin/mnist_npp \
  --images data/train-images-idx3-ubyte.gz \
  --labels data/train-labels-idx1-ubyte.gz \
  --count 5000 \
  --batch 256 \
  --threshold 80 \
  --montage_count 64 \
  --out_dir "${OUT_DIR}" | tee "${OUT_DIR}/execution.log"

echo
echo "Artifacts written to ${OUT_DIR}/"
ls -lh "${OUT_DIR}"
