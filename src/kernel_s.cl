#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX_FUNCTION_LENGTH 64

// #define DEBUG 1

typedef char BF;

__constant const static ulong POWERS_OF_2_TABLE[] = {
    1,     2,      4,      8,      16,      32,     64,    128,
    256,   512,    1024,   2048,   4096,    8192,   16384, 32768,
    65536, 131072, 262144, 524288, 1048576, 2097152};

__kernel void compute(__global int *N, __global int *result,
                      __global ulong *offset) {
  // Get our global thread ID
  ulong nf = *offset + get_global_id(0); // function number

  ulong vc = POWERS_OF_2_TABLE[*N]; // vector count

  // Our function is represented as binary number, just in reverse order
  for (ulong i = 0; i < vc; i++) {
    BF left = (nf >> i) & 1;

    BF right = (nf >> (vc - i - 1)) & 1;

    if (left == right)
      return;
  }

  // if we got here, then we have a monotonic function
  atomic_add(result, 1);
}
