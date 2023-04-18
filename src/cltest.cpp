#include <iostream>
#include <OpenCL/cl.h>
#include <fstream>

typedef unsigned long long bignum;

cl_platform_id cpPlatform; // OpenCL platform
cl_device_id device_id;    // device ID
cl_context context;        // context
cl_command_queue queue;    // command queue
cl_program program;        // program
cl_kernel kernel;          // kernel
cl_mem d_N;

std::size_t localSize = 32;

bignum runChunk(bignum offset, bignum chunkSize)
{
    bignum result = 0;
    cl_mem d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(bignum), NULL, NULL);

    cl_mem d_offset = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bignum), NULL, NULL);

    clEnqueueWriteBuffer(queue, d_offset, CL_TRUE, 0, sizeof(bignum), &offset, 0, NULL, NULL);

    auto outQueueResult = clEnqueueWriteBuffer(queue, d_result, CL_TRUE, 0, sizeof(bignum), &result, 0, NULL, NULL);

    if (outQueueResult)
    {
        printf("Error writing to result buffer\n");
        return 1;
    }

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_N);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_result);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_offset);

    std::size_t globalSize = std::pow(2, std::pow(2, N)) - localSize * localSize;

    printf("Queueing kernel with local size %lu and global size %lu\n", localSize, globalSize);

    auto kernelQueueResult = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);

    printf("Kernel queue result: %d\n", kernelQueueResult);

    if (kernelQueueResult)
    {
        char buildLog[2048];
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, globalSize, buildLog, NULL);
        printf("Error queuing kernel: %d \n", kernelQueueResult);
        return 1;
    }

    printf("Waiting for queue to finish... \n");

    clFinish(queue);

    printf("Reading result... \n");

    clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, sizeof(bignum), &result, 0, NULL, NULL);

    clReleaseMemObject(d_result);

    return result;
}

int main(int argc, char *argv[])
{

    std::ifstream kernelFile("src/kernel.cl");

    std::string src(std::istreambuf_iterator<char>(kernelFile), (std::istreambuf_iterator<char>()));

    const char *kernelSource = src.c_str();

    clGetPlatformIDs(1, &cpPlatform, NULL);

    cl_mem d_result;

    int N = 5;
    std::uint64_t result = 0;

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
        printf("Error building program: %s\n", buildLog);
        return 1;
    }

    kernel = clCreateKernel(program, "compute", &err);

    if (!kernel)
    {
        printf("Error creating kernel\n");
        return 1;
    }

    d_N = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    d_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(std::uint64_t), NULL, NULL);

    if (!d_N || !d_result)
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

    bignum functionsCount = std::pow(2, std::pow(2, N));

    bignum chunkSize = localSize * localSize;

    std::cout << "Result: " << result << std::endl;

    clReleaseMemObject(d_N);
    clReleaseMemObject(d_result);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}