// Elliott Esponda & Andrew Wheeler
// (1.) Copied
#include <cstdlib>
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <cuda.h> // (2.) Done

static const double Delta = 0.004;
static const double xMid =  0.2389;
static const double yMid =  0.55267;
static const int ThreadsPerBlock = 512; // (3.) Done

// (4.) meet fractalKernel
static __global__ void fractalKernel(const int width, const int frames, unsigned char* pic)
{
  // compute frames
  const int pixels = frames * width * width;
	const int idx = threadIdx.x + blockIdx.x * blockDim.x; 
	if(idx < pixels) // (6.) Don't use excess
	{ // (5.) Bye loops, hello constants
		const int frame = idx / (width * width);
	  const int row = (idx / width) % width;
	  const int col = idx % width;
	  //for (int frame = 0; frame < frames; frame++) { 
	  const double delta = Delta * pow(0.98, frame);
	  const double xMin = xMid - delta;
		const double yMin = yMid - delta;
	  const double dw = 2.0 * delta / width;
	  //for (int row = 0; row < width; row++) {
	  const double cy = yMin + row * dw;
	  //for (int col = 0; col < width; col++) {
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
	  pic[frame * width * width + row * width + col] = (unsigned char)depth;
	}
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
  printf("Fractal v1.7\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("computing %d frames of %d by %d fractal\n", frames, width, width);
	
	// alloc space for device copy of pic (7.)
	const int N = frames * width * width;
  unsigned char * d_pic;
  const int size = N * sizeof(unsigned char); 
	cudaMalloc((void **)&d_pic, size);

	// allocate picture array (host copies)
	unsigned char* pic = new unsigned char[N];

  // copy inputs to device
  if (cudaSuccess != cudaMemcpy(d_pic, pic, size, cudaMemcpyHostToDevice)) {fprintf(stderr, "copying to device failed\n"); exit(-1);}
	
  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  // launch GPU kernel (8.) 
  fractalKernel<<<(N + ThreadsPerBlock - 1) / ThreadsPerBlock, ThreadsPerBlock>>>(width, frames, d_pic);
  cudaDeviceSynchronize(); // (9.) Called

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);	
	CheckCuda(); // (10.) CheckCuda

	// copy result back to host
  if (cudaSuccess != cudaMemcpy(pic, d_pic, size, cudaMemcpyDeviceToHost)) {fprintf(stderr, "copying from device failed\n"); exit(-1);}
	
  // verify result by writing frames to BMP files
  if ((width <= 256) && (frames <= 100)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width, &pic[frame * width * width], name);
    }
  }

  delete [] pic;
  cudaFree(d_pic);
  return 0;
}

