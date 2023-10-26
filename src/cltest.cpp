#include <iostream>
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <fstream>
#include <cmath>
#include <climits>

typedef unsigned long long bignum;

cl_platform_id cpPlatform; // OpenCL platform
cl_device_id device_id;    // device ID
cl_context context;        // context
cl_command_queue queue;    // command queue
cl_program program;        // program
cl_kernel kernel;          // kernel
cl_mem d_N;
cl_mem d_hist;

cl_ulong hist[256];

std::size_t localSize = 32;

int result;
cl_mem d_result;
cl_mem d_offset;
bignum offset;

int dualCount;
cl_mem d_dualCount;

bignum runChunk(bignum offset, bignum chunkSize, bignum total)
{
    auto offsetResult = clEnqueueWriteBuffer(queue, d_offset, CL_TRUE, 0, sizeof(bignum), &offset, 0, NULL, NULL);

    if (offsetResult)
    {
        printf("Error writing offset result %d\n", offsetResult);
        return 1;
    }

    auto outQueueResult = clEnqueueWriteBuffer(queue, d_result, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL);

    if (outQueueResult)
    {
        printf("Error writing to result buffer\n");
        return 1;
    }

    auto out2QueueResult = clEnqueueWriteBuffer(queue, d_dualCount, CL_TRUE, 0, sizeof(int), &dualCount, 0, NULL, NULL);

    if (out2QueueResult)
    {
        printf("Error writing to result2 buffer\n");
        return 1;
    }

    auto outQueueHist = clEnqueueWriteBuffer(queue, d_hist, CL_TRUE, 0, sizeof(cl_ulong) * 256, &hist, 0, NULL, NULL);

    if (outQueueHist)
    {
        printf("Error writing to hist buffer: %d\n", outQueueHist);
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_N);
    // clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_hist);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_offset);
    clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_dualCount);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_hist);

    std::size_t globalSize = chunkSize;
    std::size_t chunkLocalSize = chunkSize;

    // printf("Queueing kernel with local size %lu and global size %lu\n", chunkLocalSize, globalSize);

    auto kernelQueueResult = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    // printf("Kernel queue result: %d\n", kernelQueueResult);

    if (kernelQueueResult)
    {
        char buildLog[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, globalSize, buildLog, NULL);
        printf("Error queuing kernel: %d \n", kernelQueueResult);
        return 1;
    }

    // printf("Waiting for queue to finish... \n");

    clFinish(queue);

    // printf("Reading result... \n");

    clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(int), &result, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_dualCount, CL_TRUE, 0, sizeof(int), &dualCount, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, d_hist, CL_TRUE, 0, sizeof(cl_ulong) * 256, &hist, 0, NULL, NULL);

    double percent = ((double)(offset + chunkSize) / (double)total) * 100;

    std::cout << "Chunk (" << offset << " , " << offset + chunkSize << ", " << percent << "%) => " << result << std::endl;

    return result;
}

int main(int argc, char *argv[])
{
#ifdef __APPLE__
    std::ifstream kernelFile("src/kernel_m.cl");
#else
    std::ifstream kernelFile("kernel.cl");
#endif

    std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

    const char *kernelSource = src.c_str();

    clGetPlatformIDs(1, &cpPlatform, NULL);

    int N = std::atoi(argv[1]);

    cl_int err = 0;

    clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    context = clCreateContext(0, 1, &device_id, NULL, NULL, NULL);

    queue = clCreateCommandQueue(context, device_id, 0, NULL);

    program = clCreateProgramWithSource(context, 1, (const char **)&kernelSource, NULL, NULL);

    auto buildError = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

    if (buildError)
    {
        char buildLog[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 2048, buildLog, NULL);
        printf("Error building program (%d): %s\n", buildError, buildLog);
        return 1;
    }

    kernel = clCreateKernel(program, "compute", &err);

    if (!kernel)
    {
        printf("Error creating kernel\n");
        return 1;
    }

    d_N = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
    d_offset = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bignum), NULL, NULL);
    d_dualCount = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int), NULL, NULL);
    d_hist = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_ulong) * 256, NULL, NULL);

    for (int i = 0; i < 256; i++)
    {
        hist[i] = 0;
    }

    if (!d_N || !d_result || !d_hist || !d_offset || !d_dualCount)
    {
        printf("Error creating buffers\n");
        return 1;
    }

    auto queueResult = clEnqueueWriteBuffer(queue, d_N, CL_TRUE, 0, sizeof(int), &N, 0, NULL, NULL);

    if (queueResult)
    {
        printf("Error writing to N buffer\n");
        return 1;
    }

    clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(localSize), &localSize, NULL);

    // localSize = 256;
    bignum functionsCount = N == 6 ? ULLONG_MAX - 1 : std::pow(2, std::pow(2, N));
    // functionsCount = functionsCount >> 1; // divide by 2
    std::cout << "funcs count: " << functionsCount << std::endl;
    bignum chunkSize = N == 6 ? 67305472 : localSize;
    if (N == 5)
        chunkSize = 16777216;
    if (N < 4)
        chunkSize = 8;
    bignum chunkableCount = functionsCount - functionsCount % chunkSize;

    bignum processed = 0;
    int chunks = 0;

    bignum percentStep = 0;
    printf("local: %d\n", localSize);
    while (processed < chunkableCount)
    {
        runChunk(processed, chunkSize, functionsCount);

        processed += chunkSize;
        chunks++;

        // float percent = ((double)processed / (double)functionsCount) * 100;

        // std::cout << "Chunk " << chunks << " => offset " << processed << "(" << percent << "%)" << std::endl;
    }

    // result++; // last function is always monotonic

    std::cout << "Result: " << result << " (" << processed << " / " << functionsCount << ")" << std::endl;
    std::cout << "Dual count out of " << result << ": " << dualCount << std::endl;

    clReleaseMemObject(d_N);
    clReleaseMemObject(d_result);
    clReleaseMemObject(d_hist);
    clReleaseMemObject(d_offset);
    clReleaseMemObject(d_dualCount);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::ofstream histFile("hist.csv");

    histFile << "INDEX,COUNT\n";
    for (int i = 0; i < 256; i++)
    {
        histFile << i + 1 << "," << hist[i] << "\n";
    }

    histFile.close();

    return 0;
}