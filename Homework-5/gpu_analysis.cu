#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
// Kernel function with parameterized block size for performance testing
__global__ void cuda_convert(uint8_t* input, uint8_t* output, int width, int height) {
    // Same kernel as previous implementation
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * 3;
        int output_idx = y * width + x;

        uint8_t r = input[idx];
        uint8_t g = input[idx + 1];
        uint8_t b = input[idx + 2];

        output[output_idx] = (uint8_t)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

// Function to run performance test
void run_performance_test(uint8_t* device_input, uint8_t* device_output, 
                           int width, int height, dim3 blockDim) {
    // Calculate grid dimensions
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);

    // Warm-up run
    cuda_convert<<<gridDim, blockDim>>>(device_input, device_output, width, height);

    // Performance measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Run kernel 200 times
    for (int i = 0; i < 200; ++i) {
        cuda_convert<<<gridDim, blockDim>>>(device_input, device_output, width, height);
    }

    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate average time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float avg_time = milliseconds / 200.0f;

    // Clean up events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Print results
    std::cout << "Block Size: " << blockDim.x << "x" << blockDim.y 
              << ", Average Execution Time: " << avg_time << " ms" << std::endl;
}

using namespace std;

int main(int argc, char *argv[]) {
    // GPU Device Query
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "Max Threads per Multiprocessor: " 
                  << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max Threads per Block: " 
                  << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Multiprocessor Count: " 
                  << deviceProp.multiProcessorCount << std::endl;
    }

    // Assume some image dimensions for testing

    // Open input file
    ifstream input_file(argv[1], ios::binary | ios::ate);
    if (!input_file) {
        cout << "Error: Could not open input file " << argv[1] << endl;
        return 1;
    }

    // Get file size and read file into host memory
    streampos file_size = input_file.tellg();
    input_file.seekg(0, ios::beg);

    vector<uint8_t> host_input(file_size);
    if (!input_file.read(reinterpret_cast<char*>(host_input.data()), file_size)) {
        cout << "Error reading input file" << endl;
        return 1;
    }
    input_file.close();
    int total_pixels = file_size / 3;
    int width = static_cast<int>(sqrt(total_pixels));
    int height = total_pixels / width;

    // Allocate device memory
    uint8_t *device_input, *device_output;
    cudaMalloc(&device_input, file_size);
    cudaMalloc(&device_output, width * height);

    // Copy input data to device
    cudaMemcpy(device_input, host_input.data(), file_size, cudaMemcpyHostToDevice);

    // Test different block sizes
    std::vector<dim3> block_sizes = {
        dim3(64, 64),    // Standard small block
         dim3(32, 16),    // Medium block
         dim3(32, 32),     // Large block
         dim3(64, 8)
    };

    // Run performance tests for each block size
    std::vector<uint8_t> host_output(width * height);

    for (const auto& blockDim : block_sizes) {
        run_performance_test(device_input, device_output, width, height, blockDim);

                //write output file

                cudaMemcpy(host_output.data(), device_output, width * height, cudaMemcpyDeviceToHost);
                std::stringstream ss;
                ss << "gc_qa" << blockDim.x << "_" << blockDim.y << ".raw";
                std::string file_name = ss.str();
                std::ofstream output_file(file_name, std::ios::binary);
                if (!output_file) {
                    std::cout << "Error: Could not open output file " << "gc_qa.raw" << std::endl;
                    return 1;
                }
        
                // Write PGM header (P5 format)
                output_file << "P5\n" << width << " " << height << "\n255\n";
                output_file.write(reinterpret_cast<char*>(host_output.data()), width * height);
                output_file.close();
    }



    // Clean up
    cudaFree(device_input);
    cudaFree(device_output);

    return 0;
}