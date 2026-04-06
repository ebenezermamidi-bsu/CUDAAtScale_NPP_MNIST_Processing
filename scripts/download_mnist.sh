#!/bin/bash

set -e

mkdir -p data
cd data

echo "Downloading MNIST dataset..."

curl -L -o train-images-idx3-ubyte.gz \
https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz

echo "Download complete."

ls -lh train-images-idx3-ubyte.gz