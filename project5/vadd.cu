#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

static const int ThreadsPerBlock = 512;

static __global__ void AddKernel(const int* a, const int* b, int* c, const int n)
{
  const int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) {
    c[idx] = a[idx] + b[idx];
  }
}

static void random_ints(int* x, const int n)
{
  for (int i = 0; i < n; i++) x[i] = i;
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[])
{
  // check command line
  if (argc != 2) {fprintf(stderr, "usage: %s number_of_elements\n", argv[0]); exit(-1);}
  const int N = atoi(argv[1]);
  if (N < 1) {fprintf(stderr, "error: number_of_elements must be at least 1\n"); exit(-1);}
  printf("running %d-element vector addition\n", N);

  // alloc space for device copies of a, b, c
  int *d_a, *d_b, *d_c;
  const int size = N * sizeof(int);
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);

  // alloc space for host copies of a, b, c and setup input values
  int* a = new int[N]; random_ints(a, N);
  int* b = new int[N]; random_ints(b, N);
  int* c = new int[N];

  // copy inputs to device
  if (cudaSuccess != cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);}
  if (cudaSuccess != cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);}

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // launch GPU kernel
  AddKernel<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(d_a, d_b, d_c, N);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.6f s\n", runtime);
  CheckCuda();

  // copy result back to host
  if (cudaSuccess != cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}

  // verify result
  for (int i = 0; i < N; i++) {
    if (c[i] != 2 * i) {
      fprintf(stderr, "error: mismatch at location %d\n", i);
      exit(-1);
    }
  }

  // cleanup
  delete [] a; delete [] b; delete [] c;
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}

