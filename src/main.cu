#include <cuda_runtime.h>
#include <npp.h>
#include <nppi.h>
#include <nppi_filtering_functions.h>
#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRows = 28;
constexpr int kCols = 28;
constexpr int kPixelsPerImage = kRows * kCols;

struct MnistImages {
    int count = 0;
    int rows = 0;
    int cols = 0;
    std::vector<uint8_t> pixels;
};

struct ImageStats {
    float mean_input = 0.0f;
    float mean_blur = 0.0f;
    float mean_edge = 0.0f;
    float max_edge = 0.0f;
    int strong_edge_pixels = 0;
};

inline void CheckCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << msg << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline void CheckNpp(NppStatus status, const char* msg) {
    if (status != NPP_SUCCESS) {
        std::ostringstream oss;
        oss << msg << " failed with NPP status " << status;
        throw std::runtime_error(oss.str());
    }
}

NppStreamContext CreateNppStreamContext(cudaStream_t stream = 0) {
    NppStreamContext ctx{};
    int device = 0;
    cudaDeviceProp prop{};

    CheckCuda(cudaGetDevice(&device), "cudaGetDevice");
    CheckCuda(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");

    unsigned int stream_flags = 0;
    CheckCuda(cudaStreamGetFlags(stream, &stream_flags), "cudaStreamGetFlags");

    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    ctx.nStreamFlags = stream_flags;

    return ctx;
}

uint32_t ReadBigEndianU32(gzFile file) {
    unsigned char buf[4];
    if (gzread(file, buf, 4) != 4) {
        throw std::runtime_error("Failed to read 4 bytes from gzip stream.");
    }
    return (static_cast<uint32_t>(buf[0]) << 24) |
           (static_cast<uint32_t>(buf[1]) << 16) |
           (static_cast<uint32_t>(buf[2]) << 8) |
           static_cast<uint32_t>(buf[3]);
}

MnistImages LoadMnistImages(const std::string& gz_path, int max_images) {
    gzFile file = gzopen(gz_path.c_str(), "rb");
    if (!file) {
        throw std::runtime_error("Could not open MNIST gzip file: " + gz_path);
    }

    const uint32_t magic = ReadBigEndianU32(file);
    if (magic != 2051) {
        gzclose(file);
        throw std::runtime_error("Invalid MNIST image file magic number.");
    }

    const int total_images = static_cast<int>(ReadBigEndianU32(file));
    const int rows = static_cast<int>(ReadBigEndianU32(file));
    const int cols = static_cast<int>(ReadBigEndianU32(file));

    if (rows != kRows || cols != kCols) {
        gzclose(file);
        throw std::runtime_error("Expected 28x28 MNIST images.");
    }

    const int count = (max_images > 0) ? std::min(max_images, total_images) : total_images;

    MnistImages result;
    result.count = count;
    result.rows = rows;
    result.cols = cols;
    result.pixels.resize(static_cast<size_t>(count) * rows * cols);

    const int bytes_to_read = count * rows * cols;
    const int bytes_read = gzread(file, result.pixels.data(), bytes_to_read);
    gzclose(file);

    if (bytes_read != bytes_to_read) {
        throw std::runtime_error("Failed to read requested MNIST image bytes.");
    }

    return result;
}

bool WritePGM(const std::string& path, const uint8_t* pixels, int rows, int cols) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
        return false;
    }
    ofs << "P5\n" << cols << " " << rows << "\n255\n";
    ofs.write(reinterpret_cast<const char*>(pixels), rows * cols);
    return static_cast<bool>(ofs);
}

__global__ void ComputeMagnitudeAndStatsKernel(
    const uint8_t* input,
    const uint8_t* blur,
    const int16_t* sobel_x,
    const int16_t* sobel_y,
    uint8_t* edge,
    int rows,
    int cols,
    int pixels_per_image,
    float* sum_input,
    float* sum_blur,
    float* sum_edge,
    float* max_edge,
    unsigned int* strong_count)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int img = blockIdx.z;

    if (x >= cols || y >= rows) {
        return;
    }

    const int idx = img * pixels_per_image + y * cols + x;

    const uint8_t in_val = input[idx];
    const uint8_t blur_val = blur[idx];
    const float gx = static_cast<float>(sobel_x[idx]);
    const float gy = static_cast<float>(sobel_y[idx]);

    float mag = sqrtf(gx * gx + gy * gy);
    if (mag > 255.0f) {
        mag = 255.0f;
    }

    const uint8_t edge_val = static_cast<uint8_t>(mag);
    edge[idx] = edge_val;

    atomicAdd(&sum_input[img], static_cast<float>(in_val));
    atomicAdd(&sum_blur[img], static_cast<float>(blur_val));
    atomicAdd(&sum_edge[img], static_cast<float>(edge_val));

    unsigned int* max_bits = reinterpret_cast<unsigned int*>(&max_edge[img]);
    atomicMax(max_bits, __float_as_uint(static_cast<float>(edge_val)));

    if (edge_val >= 100) {
        atomicAdd(&strong_count[img], 1u);
    }
}

std::string GetArgValue(int argc, char** argv, const std::string& key, const std::string& default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == key) {
            return argv[i + 1];
        }
    }
    return default_value;
}

int GetArgValueInt(int argc, char** argv, const std::string& key, int default_value) {
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == key) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_value;
}

void PrintUsage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "Options:\n"
        << "  --images <path>        Path to train-images-idx3-ubyte.gz\n"
        << "  --count <n>            Number of images to process (default: 1000)\n"
        << "  --output-dir <dir>     Output directory (default: output)\n"
        << "  --save-samples <n>     Number of sample image sets to save (default: 8)\n"
        << "  --help                 Show this help message\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--help") {
                PrintUsage(argv[0]);
                return 0;
            }
        }

        const std::string images_path =
            GetArgValue(argc, argv, "--images", "data/train-images-idx3-ubyte.gz");
        const std::string output_dir =
            GetArgValue(argc, argv, "--output-dir", "output");
        const int image_count =
            GetArgValueInt(argc, argv, "--count", 1000);
        const int save_samples =
            GetArgValueInt(argc, argv, "--save-samples", 8);

        std::cout << "Loading MNIST from " << images_path << "\n";
        MnistImages mnist = LoadMnistImages(images_path, image_count);
        std::cout << "Loaded " << mnist.count << " images of size "
                  << mnist.rows << "x" << mnist.cols << "\n";

        CheckCuda(cudaSetDevice(0), "cudaSetDevice");

        const std::string mkdir_cmd = "mkdir -p " + output_dir;
        const int mkdir_rc = std::system(mkdir_cmd.c_str());
        if (mkdir_rc != 0) {
            std::cerr << "Warning: mkdir command returned non-zero exit code.\n";
        }

        const size_t total_pixels = static_cast<size_t>(mnist.count) * kPixelsPerImage;
        const size_t bytes_u8 = total_pixels * sizeof(uint8_t);
        const size_t bytes_s16 = total_pixels * sizeof(int16_t);

        uint8_t* d_input = nullptr;
        uint8_t* d_blur = nullptr;
        int16_t* d_sobel_x = nullptr;
        int16_t* d_sobel_y = nullptr;
        uint8_t* d_edge = nullptr;

        CheckCuda(cudaMalloc(&d_input, bytes_u8), "cudaMalloc d_input");
        CheckCuda(cudaMalloc(&d_blur, bytes_u8), "cudaMalloc d_blur");
        CheckCuda(cudaMalloc(&d_sobel_x, bytes_s16), "cudaMalloc d_sobel_x");
        CheckCuda(cudaMalloc(&d_sobel_y, bytes_s16), "cudaMalloc d_sobel_y");
        CheckCuda(cudaMalloc(&d_edge, bytes_u8), "cudaMalloc d_edge");

        CheckCuda(cudaMemcpy(
            d_input,
            mnist.pixels.data(),
            bytes_u8,
            cudaMemcpyHostToDevice), "cudaMemcpy input");

        float* d_sum_input = nullptr;
        float* d_sum_blur = nullptr;
        float* d_sum_edge = nullptr;
        float* d_max_edge = nullptr;
        unsigned int* d_strong_count = nullptr;

        CheckCuda(cudaMalloc(&d_sum_input, mnist.count * sizeof(float)), "cudaMalloc d_sum_input");
        CheckCuda(cudaMalloc(&d_sum_blur, mnist.count * sizeof(float)), "cudaMalloc d_sum_blur");
        CheckCuda(cudaMalloc(&d_sum_edge, mnist.count * sizeof(float)), "cudaMalloc d_sum_edge");
        CheckCuda(cudaMalloc(&d_max_edge, mnist.count * sizeof(float)), "cudaMalloc d_max_edge");
        CheckCuda(cudaMalloc(&d_strong_count, mnist.count * sizeof(unsigned int)), "cudaMalloc d_strong_count");

        CheckCuda(cudaMemset(d_sum_input, 0, mnist.count * sizeof(float)), "cudaMemset d_sum_input");
        CheckCuda(cudaMemset(d_sum_blur, 0, mnist.count * sizeof(float)), "cudaMemset d_sum_blur");
        CheckCuda(cudaMemset(d_sum_edge, 0, mnist.count * sizeof(float)), "cudaMemset d_sum_edge");
        CheckCuda(cudaMemset(d_max_edge, 0, mnist.count * sizeof(float)), "cudaMemset d_max_edge");
        CheckCuda(cudaMemset(d_strong_count, 0, mnist.count * sizeof(unsigned int)), "cudaMemset d_strong_count");

        cudaStream_t stream = 0;
        NppStreamContext npp_ctx = CreateNppStreamContext(stream);

        const NppiSize roi = {kCols, kRows};
        const int step_u8 = kCols * static_cast<int>(sizeof(uint8_t));
        const int step_s16 = kCols * static_cast<int>(sizeof(int16_t));

        for (int i = 0; i < mnist.count; ++i) {
            const Npp8u* src = d_input + static_cast<size_t>(i) * kPixelsPerImage;
            Npp8u* blur = d_blur + static_cast<size_t>(i) * kPixelsPerImage;
            Npp16s* sobel_x = d_sobel_x + static_cast<size_t>(i) * kPixelsPerImage;
            Npp16s* sobel_y = d_sobel_y + static_cast<size_t>(i) * kPixelsPerImage;

            CheckNpp(
                nppiFilterGauss_8u_C1R_Ctx(
                    src,
                    step_u8,
                    blur,
                    step_u8,
                    roi,
                    NPP_MASK_SIZE_3_X_3,
                    npp_ctx),
                "nppiFilterGauss_8u_C1R_Ctx");

            CheckNpp(
                nppiFilterSobelHoriz_8u16s_C1R_Ctx(
                    blur,
                    step_u8,
                    sobel_x,
                    step_s16,
                    roi,
                    NPP_MASK_SIZE_3_X_3,
                    npp_ctx),
                "nppiFilterSobelHoriz_8u16s_C1R_Ctx");

            CheckNpp(
                nppiFilterSobelVert_8u16s_C1R_Ctx(
                    blur,
                    step_u8,
                    sobel_y,
                    step_s16,
                    roi,
                    NPP_MASK_SIZE_3_X_3,
                    npp_ctx),
                "nppiFilterSobelVert_8u16s_C1R_Ctx");
        }

        dim3 block(16, 16, 1);
        dim3 grid(
            (kCols + block.x - 1) / block.x,
            (kRows + block.y - 1) / block.y,
            mnist.count);

        ComputeMagnitudeAndStatsKernel<<<grid, block>>>(
            d_input,
            d_blur,
            d_sobel_x,
            d_sobel_y,
            d_edge,
            kRows,
            kCols,
            kPixelsPerImage,
            d_sum_input,
            d_sum_blur,
            d_sum_edge,
            d_max_edge,
            d_strong_count);

        CheckCuda(cudaGetLastError(), "kernel launch");
        CheckCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        std::vector<float> h_sum_input(mnist.count);
        std::vector<float> h_sum_blur(mnist.count);
        std::vector<float> h_sum_edge(mnist.count);
        std::vector<float> h_max_edge(mnist.count);
        std::vector<unsigned int> h_strong_count(mnist.count);
        std::vector<uint8_t> h_blur(total_pixels);
        std::vector<uint8_t> h_edge(total_pixels);

        CheckCuda(cudaMemcpy(h_sum_input.data(), d_sum_input, mnist.count * sizeof(float), cudaMemcpyDeviceToHost), "copy sum_input");
        CheckCuda(cudaMemcpy(h_sum_blur.data(), d_sum_blur, mnist.count * sizeof(float), cudaMemcpyDeviceToHost), "copy sum_blur");
        CheckCuda(cudaMemcpy(h_sum_edge.data(), d_sum_edge, mnist.count * sizeof(float), cudaMemcpyDeviceToHost), "copy sum_edge");
        CheckCuda(cudaMemcpy(h_max_edge.data(), d_max_edge, mnist.count * sizeof(float), cudaMemcpyDeviceToHost), "copy max_edge");
        CheckCuda(cudaMemcpy(h_strong_count.data(), d_strong_count, mnist.count * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copy strong_count");
        CheckCuda(cudaMemcpy(h_blur.data(), d_blur, bytes_u8, cudaMemcpyDeviceToHost), "copy blur");
        CheckCuda(cudaMemcpy(h_edge.data(), d_edge, bytes_u8, cudaMemcpyDeviceToHost), "copy edge");

        std::vector<ImageStats> stats(mnist.count);
        const float denom = static_cast<float>(kPixelsPerImage);
        for (int i = 0; i < mnist.count; ++i) {
            stats[i].mean_input = h_sum_input[i] / denom;
            stats[i].mean_blur = h_sum_blur[i] / denom;
            stats[i].mean_edge = h_sum_edge[i] / denom;
            stats[i].max_edge = h_max_edge[i];
            stats[i].strong_edge_pixels = static_cast<int>(h_strong_count[i]);
        }

        {
            std::ofstream csv(output_dir + "/image_stats.csv");
            if (!csv) {
                throw std::runtime_error("Could not create image_stats.csv");
            }
            csv << "image_index,mean_input,mean_blur,mean_edge,max_edge,strong_edge_pixels\n";
            csv << std::fixed << std::setprecision(4);
            for (int i = 0; i < mnist.count; ++i) {
                csv << i << ","
                    << stats[i].mean_input << ","
                    << stats[i].mean_blur << ","
                    << stats[i].mean_edge << ","
                    << stats[i].max_edge << ","
                    << stats[i].strong_edge_pixels << "\n";
            }
        }

        const int samples = std::min(save_samples, mnist.count);
        for (int i = 0; i < samples; ++i) {
            const uint8_t* input_img = mnist.pixels.data() + static_cast<size_t>(i) * kPixelsPerImage;
            const uint8_t* blur_img = h_blur.data() + static_cast<size_t>(i) * kPixelsPerImage;
            const uint8_t* edge_img = h_edge.data() + static_cast<size_t>(i) * kPixelsPerImage;

            WritePGM(output_dir + "/sample_" + std::to_string(i) + "_input.pgm", input_img, kRows, kCols);
            WritePGM(output_dir + "/sample_" + std::to_string(i) + "_blur.pgm", blur_img, kRows, kCols);
            WritePGM(output_dir + "/sample_" + std::to_string(i) + "_edge.pgm", edge_img, kRows, kCols);
        }

        {
            std::ofstream log(output_dir + "/run_log.txt");
            if (log) {
                log << "MNIST NPP processing complete\n";
                log << "Images processed: " << mnist.count << "\n";
                log << "Image size: " << kRows << "x" << kCols << "\n";
                log << "Saved samples: " << samples << "\n";
                if (!stats.empty()) {
                    log << std::fixed << std::setprecision(4);
                    log << "First image stats:\n";
                    log << "  mean_input=" << stats[0].mean_input << "\n";
                    log << "  mean_blur=" << stats[0].mean_blur << "\n";
                    log << "  mean_edge=" << stats[0].mean_edge << "\n";
                    log << "  max_edge=" << stats[0].max_edge << "\n";
                    log << "  strong_edge_pixels=" << stats[0].strong_edge_pixels << "\n";
                }
            }
        }

        std::cout << "Processing complete.\n";
        std::cout << "Artifacts written to: " << output_dir << "\n";

        cudaFree(d_input);
        cudaFree(d_blur);
        cudaFree(d_sobel_x);
        cudaFree(d_sobel_y);
        cudaFree(d_edge);
        cudaFree(d_sum_input);
        cudaFree(d_sum_blur);
        cudaFree(d_sum_edge);
        cudaFree(d_max_edge);
        cudaFree(d_strong_count);

        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "ERROR: " << ex.what() << "\n";
        return 1;
    }
}