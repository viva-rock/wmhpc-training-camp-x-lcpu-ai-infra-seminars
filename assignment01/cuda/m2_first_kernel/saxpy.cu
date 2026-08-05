#include <cuda_runtime_api.h>
#include <stdio.h>
#include <string>

#define CUDA_CHECK(expr_to_check) do {            \
    cudaError_t result  = expr_to_check;          \
    if(result != cudaSuccess)                     \
    {                                             \
        fprintf(stderr,                           \
                "CUDA Runtime Error: %s:%i:%d = %s\n", \
                __FILE__,                         \
                __LINE__,                         \
                result,\
                cudaGetErrorString(result));      \
    }                                             \
} while(0)

__global__ void saxpy (float *x, float *y, int n)
{
    for(int idx = threadIdx.x + blockIdx.x * blockDim.x;
        idx < n; idx += blockDim.x * gridDim.x)
        y[idx] += 2 * x[idx];
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("The number of params isn't correct./n");
        return 1;
    }

    int n = std::stoi(argv[1]);
    int bytes = n * sizeof(float);
    if(n == 0)
    {
        printf("SUM=0\n");
        return 0;
    }

    float* x = (float*)malloc(bytes);
    float* y = (float*)malloc(bytes);
    for(int i=0; i<n; i++)
    {
        x[i] = ((i%2048) - 1024) * 0.5f;
        y[i] = (i%1024) - 512;
    }

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));

    int start_time = (int)clock();
    CUDA_CHECK(cudaMemcpy(d_x, x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y, bytes, cudaMemcpyHostToDevice));

    int thread_num = 1024;
    int block_num = (n + 1023) / 1024;
    saxpy<<<block_num, thread_num>>>(d_x, d_y, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(y, d_y, bytes, cudaMemcpyDeviceToHost));

    int total_time = (int)clock() - start_time;
    double sum=0;
    for(int i=0; i<n; i++) sum += y[i];
    printf("SUM=%.0f\n TIME=%d", sum, total_time);

    return 0;
}