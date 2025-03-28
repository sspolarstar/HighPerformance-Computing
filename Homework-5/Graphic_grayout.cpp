// this file will take a input .raw of size 1024x1024 in 24bit RGB. Then it will convert
// it to grayscale.

// it must use cuda to implement. 
// does the following:
// read the image file
// write a kernel function to perform the grey scale conversion
// save the converted image into a binary file named "gc.raw"

// for part 2:
    //fold report for question -
            // [u1203152@notch271:grayscale]$ ./device_query.o 
            // ./device_query.o Starting...

            //  CUDA Device Query (Runtime API) version (CUDART static linking)

            // Detected 1 CUDA Capable device(s)

            // Device 0: "NVIDIA GeForce RTX 2080 Ti"
            //   CUDA Driver Version / Runtime Version          12.8 / 11.6
            //   CUDA Capability Major/Minor version number:    7.5
            //   Total amount of global memory:                 10823 MBytes (11348672512 bytes)
        //   (068) Multiprocessors, (064) CUDA Cores/MP:    4352 CUDA Cores
            //   GPU Max Clock rate:                            1545 MHz (1.54 GHz)
            //   Memory Clock rate:                             7000 Mhz
            //   Memory Bus Width:                              352-bit
            //   L2 Cache Size:                                 5767168 bytes
            //   Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
            //   Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
            //   Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
            //   Total amount of constant memory:               65536 bytes
            //   Total amount of shared memory per block:       49152 bytes
        //   Total shared memory per multiprocessor:        65536 bytes
            //   Total number of registers available per block: 65536
            //   Warp size:                                     32
        //   Maximum number of threads per multiprocessor:  1024
            //   Maximum number of threads per block:           1024
            //   Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
            //   Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
            //   Maximum memory pitch:                          2147483647 bytes
            //   Texture alignment:                             512 bytes
            //   Concurrent copy and kernel execution:          Yes with 3 copy engine(s)
            //   Run time limit on kernels:                     No
            //   Integrated GPU sharing Host Memory:            No
            //   Support host page-locked memory mapping:       Yes
            //   Alignment requirement for Surfaces:            Yes
            //   Device has ECC support:                        Disabled
            //   Device supports Unified Addressing (UVA):      Yes
            //   Device supports Managed Memory:                Yes
            //   Device supports Compute Preemption:            Yes
            //   Supports Cooperative Kernel Launch:            Yes
            //   Supports MultiDevice Co-op Kernel Launch:      Yes
            //   Device PCI Domain ID / Bus ID / location ID:   0 / 61 / 0
            //   Compute Mode:
            //      < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

            // deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.8, CUDA Runtime Version = 11.6, NumDevs = 1
            // Result = PASS

    //part 2 has a few parts,
    // but the reported information includes a bit.
    // Max number of threads per SM = maximum number of threads per streaming multiprocessor
        //which comes out to 1024.
    //With a max number of 8 blocks, that's  1024/8 = 128 threads per block. 

//part 2 (b) 
    // Choose 3 block sizes.
    // I choose 16x16, 16x32, 32x32. 
        //For 16x16 blocks (256 threads)
            //under utilization of GPU.
            //fine for small images.
        //for 16x32 blocks (512 threads)
            //about half utilization.
            //should be significantly better.
        //for 32x32 (1024 threads)
            // best utilization, but more time, likely additional over head/bad allocation timeing?
    //Output timing report:
        // Block Size: 16x16, Average Execution Time: 0.0110509 ms
        // Block Size: 32x16, Average Execution Time: 0.01028 ms
        // Block Size: 32x32, Average Execution Time: 0.0121958 ms
    //For fun I tried a block size much larger (64x64) and the Average Execution Time: 0.00014864 ms. 
    // although no image was produced, so this is clearly an error that was invisible to my testing. 

    #include <cuda_runtime.h>
    #include <iostream>
    #include <fstream>
    #include <vector>
    #include <cmath>
    #include <cstdint>
    
    using namespace std;
    
    // Kernel function to convert image to grayscale
    __global__ void cuda_convert(uint8_t* input, uint8_t* output, int width, int height) {
        // Calculate global thread index
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
    
        // Check if thread is within image bounds
        if (x < width && y < height) {
            // Calculate linear index for the input (3 channels per pixel)
            int idx = (y * width + x) * 3;
            int output_idx = y * width + x;
            // get RGB values
            uint8_t r = input[idx];
            uint8_t g = input[idx + 1];
            uint8_t b = input[idx + 2];
    
            // convert to grayscale using weighted method (NTSC formula)
            output[output_idx] = static_cast<uint8_t>(0.299f * r + 0.587f * g + 0.114f * b);
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
    
        // Get file size and read file into host memory
        streampos file_size = input_file.tellg();
        input_file.seekg(0, ios::beg);
    
        vector<uint8_t> host_input(file_size);
        if (!input_file.read(reinterpret_cast<char*>(host_input.data()), file_size)) {
            cout << "Error reading input file" << endl;
            return 1;
        }
        input_file.close();
    
        // Assume input is 24-bit RGB (3 bytes per pixel)
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
        cudaDeviceSynchronize();
    
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
            return 1;
        }
    
        // Allocate host memory for output
        vector<uint8_t> host_output(width * height);
        cudaMemcpy(host_output.data(), device_output, width * height, cudaMemcpyDeviceToHost);
    
        // Write output file with a simple PGM header for grayscale images
        ofstream output_file(argv[2], ios::binary);
        if (!output_file) {
            cout << "Error: Could not open output file " << argv[2] << endl;
            return 1;
        }
    
        // Write PGM header (P5 format)
        output_file << "P5\n" << width << " " << height << "\n255\n";
        output_file.write(reinterpret_cast<char*>(host_output.data()), width * height);
        output_file.close();
    
        // Free device memory
        cudaFree(device_input);
        cudaFree(device_output);
    
        cout << "Grayscale conversion complete. Output saved to " << argv[2] << endl;
        return 0;
    }
    