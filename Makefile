CUDA_HOME ?= /usr/local/cuda
NVCC := $(CUDA_HOME)/bin/nvcc

TARGET := bin/mnist_npp
SRC := src/main.cu

NVCCFLAGS := -O3 -std=c++17 -arch=native -Xcompiler -Wall,-Wextra
LDFLAGS := -lnppif -lnppc -lcudart -lz

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)
	rm -rf output