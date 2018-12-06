/*
Fractal code for CS 4380 / CS 5351

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
#include <cmath>
#include <cuda.h>

static const int ThreadsPerBlock = 512;

static const double Delta = 0.004;
static const double xMid =  0.2389;
static const double yMid =  0.55267;

static __global__ void fractalKernel(const int start_frame, const int gpu_frames, const int width, unsigned char* pic_d)
{
  // todo: use the GPU to compute the frames (base the code on the previous project)
  // compute frames
  const int idx = threadIdx.x + blockIdx.x * blockDim.x + (start_frame * width * width);
  if(idx < (start_frame + gpu_frames) * width * width)
  { 
    const int frame = idx / (width * width);
    const int row = (idx / width) % width;
    const int col = idx % width;
    const double delta = Delta * pow(0.98, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    const double cy = yMin + row * dw;
    const double cx = xMin + col * dw;
    double x = cx;
    double y = cy;
    int depth = 256;
    double x2, y2;
    do {
      x2 = x * x;
      y2 = y * y;
      y = 2 * x * y + cy;
      x = x2 - y2 + cx;
      depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0));
    pic_d[(frame-start_frame) * width * width + row * width + col] = (unsigned char)depth;
  }
}

unsigned char* GPU_Init(const int gpu_frames, const int width)
{
  unsigned char* pic_d;
  if (cudaSuccess != cudaMalloc((void **)&pic_d, gpu_frames * width * width * sizeof(unsigned char))) {fprintf(stderr, "could not allocate memory\n"); exit(-1);}
  return pic_d;
}

void GPU_Exec(const int start_frame, const int gpu_frames, const int width, unsigned char* pic_d)
{
  // todo: launch the kernel with ThreadsPerBlock and the appropriate number of blocks
  fractalKernel<<<((gpu_frames * width * width) + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(start_frame, gpu_frames, width, pic_d);
}

void GPU_Fini(const int gpu_frames, const int width, unsigned char* pic, unsigned char* pic_d)
{
  // todo: copy the result from the device to the host and free the device memory
  const int size = gpu_frames * width * width * sizeof(unsigned char);
  if (cudaSuccess != cudaMemcpy(pic, pic_d, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}
  cudaFree(pic_d);
}
