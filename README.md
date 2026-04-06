# GPU-Accelerated MNIST Edge Pipeline with CUDA NPP

This project uses NVIDIA CUDA and the NPP library to process a large batch of MNIST handwritten digit images on the GPU. The application loads thousands of small grayscale images, applies Gaussian smoothing with NPP, computes Sobel gradients with NPP, and then launches a custom CUDA kernel to calculate edge magnitude and per-image statistics.

The goal was to demonstrate practical GPU-based image processing on a dataset large enough to meet the assignment requirements.

---

## Use case

A practical use case is OCR and document preprocessing at scale. Before handwritten characters are classified or fed into a recognition pipeline, it is common to denoise, smooth, and extract edges. This project demonstrates that workflow on the MNIST dataset.

The pipeline is:

1. Read MNIST images from the original `.gz` IDX file  
2. Transfer images to GPU memory  
3. Use CUDA NPP to apply:  
   - Gaussian blur  
   - Sobel horizontal gradient  
   - Sobel vertical gradient  
4. Use a custom CUDA kernel to:
   - compute edge magnitude  
   - calculate per-image statistics  
5. Write results to disk:
   - CSV statistics  
   - logs  
   - sample processed images  

---

## What this project demonstrates

- GPU-accelerated image processing using CUDA  
- Use of NVIDIA NPP (high-performance GPU library)  
- Custom CUDA kernel for post-processing  
- Batch processing of thousands of small images  
- End-to-end pipeline from input data → GPU → output artifacts  

---

## Dataset

This project uses the MNIST handwritten digit dataset.

- 70,000 grayscale images  
- Each image is 28×28 pixels  
- Data is stored in compressed IDX format (`.gz`)  

The program processes thousands of images in a single execution (e.g., 5000 images).

Dataset is downloaded using:

    bash scripts/download_mnist.sh

---

## GPU processing details

This project uses both:

### CUDA NPP (library)
- Gaussian filtering  
- Sobel edge detection  

### Custom CUDA kernel
- Combines Sobel X and Y gradients  
- Computes edge magnitude  
- Calculates statistics per image  

---

## Repository layout

.
├── Makefile  
├── README.md  
├── run.sh  
├── scripts/download_mnist.sh  
├── src/main.cu  
├── data  
└── output (generated)  

---

## Build

    make

---

## Run

    bash scripts/download_mnist.sh
    bash run.sh

Manual run:

    ./mnist_npp --images data/train-images-idx3-ubyte.gz --count 5000 --output-dir output

---

## Output artifacts

After running, the output/ directory contains:

- image_stats.csv  
- run_log.txt  
- execution.log  
- sample_*_input.pgm  
- sample_*_blur.pgm  
- sample_*_edge.pgm  

---

## Example run

    Loaded 5000 images of size 28x28
    Processing complete.
    Artifacts written to: output

---

## Lessons learned

Setting up CUDA, WSL, and NPP required careful configuration. Adapting to newer NPP APIs and ensuring compatibility between toolkit and driver was also a key learning.

---

## Suggested proof

- CSV output  
- logs  
- sample images  
- terminal screenshot  
