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

#ifdef DEBUG
  if (nf != 11)
    return;
  printf("nf = %lu, N = %d\n", nf, *N);
#endif

  ulong vc = POWERS_OF_2_TABLE[*N]; // vector count

#ifdef DEBUG
  printf("vc = %lu", vc);
#endif

  BF f[MAX_FUNCTION_LENGTH];  // base function
  BF fd[MAX_FUNCTION_LENGTH]; // direct transform
  BF fi[MAX_FUNCTION_LENGTH]; // inverse transform
  BF fr[MAX_FUNCTION_LENGTH]; // result function

  int energy = 0; // energy determines the criteria for the function

  bool isRightmostValueZero =
      false; // quick criteria for monotonicity: if f[0] is 1 and any of f[i >
             // 0] = 0 then function is not monotonic
  bool isLeftmostValueOne = false;

  // Our function is represented as binary number, just in reverse order
  for (ulong i = 0; i < vc; i++) {
    BF value = (nf >> i) & 1;

    int fidx = vc - i - 1;

    f[fidx] = value;

    if (i == 0 && value == 0) {
      isRightmostValueZero = true;
    }

    if (i == vc - 1 && value == 1) {
      isLeftmostValueOne = true;
    }

    // copy the values right away
    fd[fidx] = f[fidx];
    fi[fidx] = f[fidx];

    // quick monotonicity checks
    if (i > 0 && value == 1 && isRightmostValueZero)
      return;
    if (i < vc - 1 && value == 0 && isLeftmostValueOne)
      return;
  }

#ifdef DEBUG
  printf("f = ");
  for (int i = 0; i < vc; i++) {
    printf("%d ", f[i]);
  }
  printf("\n");
#endif

  ////////////

#ifdef DEBUG
  printf("Perf with %d, %d\n", n);
#endif

  int n = *N; // amount of variables or size of function

  ulong vhalf = POWERS_OF_2_TABLE[n - 1]; // half of loop

  ulong npower = POWERS_OF_2_TABLE[n]; // power of 2 for n

#ifdef DEBUG
  printf("vhalf = %ul, npower = %ul, n = %d, PT[n] = %d\n", vhalf, npower, n,
         POWERS_OF_2_TABLE[n]);
#endif

  BF fo[MAX_FUNCTION_LENGTH];  // buffer for direct
  BF foi[MAX_FUNCTION_LENGTH]; // buffer for inverse

  for (int i = 0; i < n; i++) {
    // copy last iteration
    for (int j = 0; j < vc; j++) {
      fo[j] = fd[j];
      foi[j] = fi[j];
    }
    for (int j = 0; j < vc; j++) {
#ifdef DEBUG
      printf("Iteration with n = %d, vhalf = %d, npower = %d, si = %d\n", n,
             vhalf, npower, si);
#endif
      int jd = j << 1; // j * 2

      // according to formulas
      if (j < vhalf) {
        fd[j] = fo[jd];
        fi[j] = foi[jd] - foi[jd + 1];
      } else {
        fd[j] = fo[jd - npower] + fo[jd - npower + 1];
        fi[j] = foi[jd - npower + 1];
      }
    }
  }

  ///////////

#ifdef DEBUG
  printf("fd = ");
  for (int i = 0; i < vc; i++) {
    printf("%d ", fd[i]);
  }
  printf("\n");
#endif DEBUG

#ifdef DEBUG
  printf("fi = ");
  for (int i = 0; i < vc; i++) {
    printf("%d ", fi[i]);
  }
  printf("\n");
#endif

  // actual criteria check
  // return as soon as we find a violation
  for (int i = 0; i < vc; i++) {
    fr[i] = fd[i] * fi[i];
    energy += f[i] * f[i];

    if (i == vc - 1) {
      if (abs(fr[i] - energy) > 0.01) {
        return;
      }
    } else {
      if (abs(fr[i]) > 0.01) { // this works inside loop just because we do not
                               // use energy variable
        return;
      }
    }
  }

#ifdef DEBUG
  printf("fr = ");
  for (int i = 0; i < vc; i++) {
    printf("%d ", fr[i]);
  }
  printf("\n");

  printf("energy = %d\n", energy);
#endif

  // if we got here, then we have a monotonic function
  atomic_add(result, 1);
}
