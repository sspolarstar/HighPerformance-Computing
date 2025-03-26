#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string.h>

using namespace std;

// Kernel function to convert image to grayscale
__global__ void cuda_convert(uint8_t* input, uint8_t* output, int width, int height) {
    // Calculate global thread index
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if thread is within image bounds
    if (x < width && y < height) {
        // Calculate linear index
        int idx = (y * width + x) * 3;
        int output_idx = y * width + x;

        // Extract RGB values
        uint8_t r = input[idx];
        uint8_t g = input[idx + 1];
        uint8_t b = input[idx + 2];

        // Convert to grayscale using weighted method
        output[output_idx] = (uint8_t)(0.21f * r + 0.72f * g + 0.07f * b);
    }
}

int main(int argc, char *argv[]) {
    // Check command line arguments
    if (argc != 3) {
        cout << "Usage: ./Graphic_grayout <input_file> <output_file>" << endl;
        return 1;
    }

    // Open input file
    ifstream input_file(argv[1], ios::binary | ios::ate);
    if (!input_file) {
        cout << "Error: Could not open input file " << argv[1] << endl;
        return 1;
    }

    // Get file size
    streampos file_size = input_file.tellg();
    input_file.seekg(0, ios::beg);

    // Read file into host memory
    vector<uint8_t> host_input(file_size);
    if (!input_file.read(reinterpret_cast<char*>(host_input.data()), file_size)) {
        cout << "Error reading input file" << endl;
        return 1;
    }
    input_file.close();

    // Assume input is 24-bit RGB (3 bytes per pixel)
    // Calculate width and height (this is a simplification - you may need to pass these)
    int total_pixels = file_size / 3;
    int width = static_cast<int>(sqrt(total_pixels));
    int height = total_pixels / width;

    // Allocate device memory
    uint8_t *device_input, *device_output;
    cudaMalloc(&device_input, file_size);
    cudaMalloc(&device_output, width * height);

    // Copy input data to device
    cudaMemcpy(device_input, host_input.data(), file_size, cudaMemcpyHostToDevice);

    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y);

    // Launch kernel
    cuda_convert<<<gridDim, blockDim>>>(device_input, device_output, width, height);

    // Synchronize device
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Allocate host memory for output
    vector<uint8_t> host_output(width * height);

    // Copy results back to host
    cudaMemcpy(host_output.data(), device_output, width * height, cudaMemcpyDeviceToHost);

    // Write output file
    ofstream output_file(argv[2], ios::binary);
    if (!output_file) {
        cout << "Error: Could not open output file " << argv[2] << endl;
        return 1;
    }
    output_file.write(reinterpret_cast<char*>(host_output.data()), width * height);
    output_file.close();

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);

    cout << "Grayscale conversion complete. Output saved to " << argv[2] << endl;

    return 0;
}