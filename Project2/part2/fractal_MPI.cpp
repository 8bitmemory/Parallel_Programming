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
#include <cmath>
#include <sys/time.h>
#include "cs43805351.h"
#include <mpi.h>

static const double Delta = 0.004;
static const double xMid =  0.2389;
static const double yMid =  0.55267;

static void fractal(const int width, const int frames, unsigned char* pic, int beg, int end)
{	
 // compute frames
  for (int frame = beg; frame < end; frame++) {
    const double delta = Delta * pow(0.98, frame);
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {
      const double cy = yMin + row * dw;
      for (int col = 0; col < width; col++) {
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
        pic[(frame-beg) * width * width + row * width + col] = (unsigned char)depth;
      }
    }
  }
}

int main(int argc, char *argv[])
{
  int comm_sz, my_rank;
  int my_beg;
  int my_end;

  MPI_Init(NULL, NULL);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  printf("Fractal v1.7\n");

  // check command line
  if (argc != 3) {fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]); exit(-1);}
  const int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: frame_width must be at least 10\n"); exit(-1);}
  const int frames = atoi(argv[2]);
  if (frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("computing %d frames of %d by %d fractal\n", frames, width, width);
  
  my_beg = my_rank * frames / comm_sz;
  my_end = (my_rank + 1) * frames / comm_sz;

  // allocate picture array
  if(my_rank == 0)
  unsigned char* pic = new unsigned char[frames * width * width];

  unsigned char * pic1_buff = new unsigned char[(frames / comm_sz) * width * width];

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  fractal(width, frames, pic, my_beg, my_end);

  //gather call
  MPI_Gather(pic1_buff,(end-beg), MPI_UNSIGNED_CHAR,pic, ((end-beg) * width * width), MPI_UNSIGNED_CHAR,0,MPI_COMM_WORLD);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);

  // verify result by writing frames to BMP files
  if ((width <= 256) && (frames <= 100)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width, &pic[frame * width * width], name);
    }
  }

  delete [] pic;
  return 0;
}

