/*
Collatz code for CS 4380 / CS 5351

Copyright (c) 2018, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is *not* permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static __global__ void collatzKernel(int * range, int * maxlen)
{
  // compute sequence lengths
  const long idx = threadIdx.x + blockIdx.x * (long)blockDim.x;
  long val = idx+1;
  int len = 1;

  if(idx < *range)
  while (val != 1) {
    len++;
    if ((val % 2) == 0) {
      val = val / 2;  // even
    } else {
      val = 3 * val + 1;  // odd
    }
  }
  if (*maxlen < len)atomicMax(maxlen,len);

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
  printf("Collatz v1.0\n");

  // check command line
  if (argc != 2) {fprintf(stderr, "usage: %s range\n", argv[0]); exit(-1);}
  const int range = atol(argv[1]);
  if (range < 1) {fprintf(stderr, "error: range must be at least 1\n"); exit(-1);}
  printf("range: 1, ..., %ld\n", range);

  
  // alloc space for device copies of maxlen and range
  int* d_maxlen;
  int* d_range;
  const int size = sizeof(int);
  cudaMalloc((void **)&d_maxlen, size);
  cudaMalloc((void **)&d_range, size);


  // alloc space for host copies of a, b, c and setup input values
  int* h_maxlen = new int;
  int* h_range = new int;
  *d_maxlen = 0;
  h_range = &range;

  // copy inputs to device
  if (cudaSuccess != cudaMemcpy(d_maxlen, h_maxlen, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);};
  if (cudaSuccess != cudaMemcpy(d_range, h_range, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);};
  

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  //launch GPU kernel
  collatzKernel<<<(ThreadsPerBlock + range - 1)/ThreadsPerBlock,ThreadsPerBlock>>>(d_range, d_maxlen);
  cudaDeviceSynchronize();

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);
  CheckCuda();

  // copy result back to host
  if (cudaSuccess != cudaMemcpy(h_maxlen, d_maxlen, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}

  // print result
  printf("longest sequence: %d elements\n", *h_maxlen);

  // clean up
  delete h_maxlen;
  delete h_range;
  cudaFree(d_maxlen);
  cudaFree(d_range);

  return 0;
}

