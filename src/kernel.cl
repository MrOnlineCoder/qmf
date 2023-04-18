#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable

#define MAX_FUNCTION_LENGTH 512

#define DEBUG 0

void _directQuickTransform(int *f, int sz, int si, int n)
{
    ulong vhalf = pown(2.f, n) / 2;

    ulong npower = pown(2.f, n);

    if (DEBUG)
        printf("Direct with n = %d, vhalf = %d, npower = %d, si = %d\n", n, vhalf, npower, si);

    int fo[MAX_FUNCTION_LENGTH];

    for (int i = 0; i < sz; i++)
    {
        fo[i] = f[i];
    }

    for (int i = 0; i < sz; i++)
    {
        if (i < vhalf)
        {
            if (si == 0)
            {
                f[i] = fo[2 * i] + fo[2 * i - npower + 1];
            }
            else
            {
                f[i] = fo[2 * i];
            }
        }
        else
        {
            if (si == 0)
            {
                f[i] = fo[2 * i - npower + 1];
            }
            else
            {
                f[i] = fo[2 * i - npower] + fo[2 * i - npower + 1];
            }
        }
    }
}

void _inverseQuickTransform(int *f, int sz, int si, int n)
{
    ulong vhalf = pown(2.f, n) / 2;

    ulong npower = pown(2.f, n);

    if (DEBUG)
        printf("Inverse with n = %d, vhalf = %d, npower = %d, si = %d\n", n, vhalf, npower, si);

    int fo[MAX_FUNCTION_LENGTH];

    for (int i = 0; i < sz; i++)
    {
        fo[i] = f[i];
    }

    for (int i = 0; i < sz; i++)
    {
        if (i < vhalf)
        {
            if (si == 0)
            {
                f[i] = fo[2 * i];
            }
            else
            {
                f[i] = fo[2 * i] - fo[2 * i + 1];
            }
        }
        else
        {
            if (si == 0)
            {
                f[i] = fo[2 * i - npower + 1] - fo[2 * i - npower];
            }
            else
            {
                f[i] = fo[2 * i - npower + 1];
            }
        }
    }
}

void performQuickTransform(int *f, int sz, int n, bool inverse)
{
    for (int i = 0; i < n; i++)
    {
        int si = 1;
        if (inverse)
        {
            _directQuickTransform(f, sz, si, n);
        }
        else
        {
            _inverseQuickTransform(f, sz, si, n);
        }
    }
}

__kernel void compute(__global int *N,
                      __global int *result)
{
    // Get our global thread ID
    ulong nf = get_global_id(0);

    if (DEBUG)
        printf("nf = %lu, N = %d\n", nf, *N);

    ulong vc = pown(2.f, *N);
    ulong fc = pown(2.f, vc);

    if (DEBUG)
        printf("vc = %lu, fc = %lu\n", vc, fc);

    int f[MAX_FUNCTION_LENGTH];
    int fd[MAX_FUNCTION_LENGTH];
    int fi[MAX_FUNCTION_LENGTH];
    int fr[MAX_FUNCTION_LENGTH];

    int energy = 0;

    for (int i = 0; i < vc; i++)
    {
        f[vc - i - 1] = (nf >> i) & 1;
    }

    if (DEBUG)
    {
        printf("f = ");
        for (int i = 0; i < vc; i++)
        {
            printf("%d ", f[i]);
        }
        printf("\n");
    }

    for (int i = 0; i < vc; i++)
    {
        fd[i] = f[i];
        fi[i] = f[i];
    }

    performQuickTransform(&fd, vc, *N, false);
    performQuickTransform(&fi, vc, *N, true);

    if (DEBUG)
    {
        printf("fd = ");
        for (int i = 0; i < vc; i++)
        {
            printf("%d ", fd[i]);
        }
        printf("\n");
    }

    if (DEBUG)
    {
        printf("fi = ");
        for (int i = 0; i < vc; i++)
        {
            printf("%d ", fi[i]);
        }
        printf("\n");
    }

    for (int i = 0; i < vc; i++)
    {
        fr[i] = fd[i] * fi[i];
        energy += f[i] * f[i];
    }

    if (DEBUG)
    {
        printf("fr = ");
        for (int i = 0; i < vc; i++)
        {
            printf("%d ", fr[i]);
        }
        printf("\n");
    }

    if (DEBUG)
        printf("energy = %d\n", energy);

    volatile int criteria = 1;

    for (int i = 0; i < vc; i++)
    {
        if (i == vc - 1)
        {
            if (abs(fr[i] - energy) > 0.01)
            {
                criteria = 0;
                break;
            }
        }
        else
        {
            if (abs(fr[i]) > 0.01)
            {
                criteria = 0;
                break;
            }
        }
    }

    if (criteria == 1)
    {
        atomic_add(result, 1);
    }
}
