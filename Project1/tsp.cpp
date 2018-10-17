/*
Genetic algorithm based TSP code for CS 4380 / CS 5351

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
#include <cstring>
#include <cmath>
#include <climits>
#include <algorithm>
#include <sys/time.h>
#include "cs43805351.h"

static inline int dist(const int a, const int b, const float px[], const float py[])
{
  return (int)(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])) + 0.5f);
}

static int tourLength(const int cities, const int tour[], const float px[], const float py[])
{
  int len = dist(tour[cities - 1], tour[0], px, py);
  for (int j = 1; j < cities; j++) {
    len += dist(tour[j - 1], tour[j], px, py);
  }
  return len;
}

static inline unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static int pickParent(const int pop, const float range[], const int seed)
{
  float rnd = 1.0f * hash(seed) / UINT_MAX;
  int p = 0;
  while ((p < pop - 1) && (rnd > range[p])) {
    rnd -= range[p];
    p++;
  }
  return p;
}

static void drawTour(const int cities, const float posx[], const float posy[], int tour[])
{
  const int width = 512;
  unsigned char pic[width][width];
  memset(pic, 0, width * width * sizeof(unsigned char));
  float minx = posx[0], maxx = posx[0];
  float miny = posy[0], maxy = posy[0];
  for (int i = 1; i < cities; i++) {
    if (minx > posx[i]) minx = posx[i];
    if (maxx < posx[i]) maxx = posx[i];
    if (miny > posy[i]) miny = posy[i];
    if (maxy < posy[i]) maxy = posy[i];
  }
  const float distx = maxx - minx;
  const float disty = maxy - miny;
  const float factorx = (width - 1) / distx;
  const float factory = (width - 1) / disty;
  int x[cities], y[cities];
  for (int i = 0; i < cities; i++) {
    x[i] = (int)(0.5f + (posx[tour[i]] - minx) * factorx);
    y[i] = (int)(0.5f + (posy[tour[i]] - miny) * factory);
  }
  for (int i = 1; i < cities; i++) {
    line(x[i - 1], y[i - 1], x[i], y[i], 127, (unsigned char*)pic, width);
  }
  line(x[cities - 1], y[cities - 1], x[0], y[0], 128, (unsigned char*)pic, width);
  for (int i = 0; i < cities; i++) {
    line(x[i], y[i], x[i], y[i], 255, (unsigned char*)pic, width);
  }
  writeBMP(width, width, (unsigned char*)pic, "tsp.bmp");
}

static int tsp(const int cities, const int pop, const int generations, const float px[], const float py[], int besttour[])
{
  // allocate memory
  float range[pop];
  int length[pop], *tour[pop], *tour2[pop];
  for (int i = 0; i < pop; i++) {
    tour[i] = new int[cities];
    tour2[i] = new int[cities];
  }

  // initialize tours
  for (int i = 0; i < pop; i++) {
    for (int j = 0; j < cities; j++) {
      tour[i][j] = j;
    }
  }

  // randomize tours
  for (int i = 0; i < pop; i++) {
    for (int j = 0; j < cities; j++) {
      const int seed = (i * cities + j) * -2;
      const int pos1 = hash(seed) % cities;
      const int pos2 = hash(seed - 1) % cities;
      std::swap(tour[i][pos1], tour[i][pos2]);
    }
  }

  // compute tour lengths
  int best = 0, worst = 0;
  for (int i = 0; i < pop; i++) {
    length[i] = tourLength(cities, tour[i], px, py);
    if (length[best] > length[i]) best = i;
    if (length[i] > length[worst]) worst = i;
  }

  // run generations
  for (int gen = 1; gen < generations; gen++) {
    // compute range for finding parents based on fitness
    const float wlength = length[worst];
    for (int i = 0; i < pop; i++) range[i] = wlength / length[i];
    float rsum = range[0];
    for (int i = 1; i < pop; i++) rsum += range[i];
    const float irsum = 1.0f / rsum;
    for (int i = 0; i < pop; i++) range[i] *= irsum;

    // keep the best
    for (int j = 0; j < cities; j++) tour2[0][j] = tour[best][j];

    // mutate
    for (int i = 1; i < pop / 2; i++) {
      const int seed = (gen * pop + i) * 4;
      const int parent = pickParent(pop, range, seed);
      for (int j = 0; j < cities; j++) tour2[i][j] = tour[parent][j];
      int pos1 = hash(seed + 1) % cities;
      int pos2 = hash(seed + 2) % cities;
      std::swap(tour2[i][pos1], tour2[i][pos2]);
    }

    // cross-over
    for (int i = pop / 2; i < pop; i++) {
      const int seed = (gen * pop + i) * 4;
      const int parent1 = pickParent(pop, range, seed);
      const int parent2 = pickParent(pop, range, seed + 1);
      const int pos1 = hash(seed + 2) % cities;
      const int pos2 = hash(seed + 3) % cities;
      bool used[cities];
      for (int j = 0; j < cities; j++) used[j] = false;
      for (int j = pos1; j != pos2; j = (j + 1) % cities) {
        const int city = tour[parent1][j];
        tour2[i][j] = city;
        used[city] = true;
      }
      int pos = pos2;
      for (int j = 0; j < cities; j++) {
        const int city = tour[parent2][j];
        if (!used[city]) {
          tour2[i][pos] = city;
          pos = (pos + 1) % cities;
        }
      }
    }

    // exchange old and new generation
    for (int i = 0; i < pop; i++) {
      std::swap(tour[i], tour2[i]);
    }

    // compute tour lengths
    const int old = length[best];
    best = 0; worst = 0;
    for (int i = 0; i < pop; i++) {
      length[i] = tourLength(cities, tour[i], px, py);
      if (length[best] > length[i]) best = i;
      if (length[i] > length[worst]) worst = i;
    }
  }

  // return best tour
  for (int j = 0; j < cities; j++) besttour[j] = tour[best][j];

  // free memory
  for (int i = 0; i < pop; i++) delete [] tour2[i];
  for (int i = 0; i < pop; i++) delete [] tour[i];

  return length[best];
}

int main(int argc, char *argv[])
{
  printf("TSP v1.5\n");

  // read input
  if (argc != 4) {fprintf(stderr, "usage: %s input_file population_size number_of_generations\n", argv[0]); exit(-1);}
  FILE* f = fopen(argv[1], "rb");
  if (f == NULL) {fprintf(stderr, "error: could not open file %s\n", argv[1]); exit(-1);}
  int cities;
  int cnt = fread(&cities, sizeof(int), 1, f);
  if (cnt != 1) {fprintf(stderr, "error: failed to read cities\n"); exit(-1);}
  if (cities < 1) {fprintf(stderr, "error: cities must be at least 1\n"); exit(-1);}

  float posx[cities], posy[cities];
  cnt = fread(posx, sizeof(float), cities, f);
  if (cnt != cities) {fprintf(stderr, "error: failed to read posx\n"); exit(-1);}
  cnt = fread(posy, sizeof(float), cities, f);
  if (cnt != cities) {fprintf(stderr, "error: failed to read posy\n"); exit(-1);}
  fclose(f);

  const int popsize = atoi(argv[2]);
  if (popsize < 4) {fprintf(stderr, "error: population size must be at least 4\n"); exit(-1);}
  const int generations = atoi(argv[3]);
  if (generations < 1) {fprintf(stderr, "error: number of generations must be at least 1\n"); exit(-1);}

  printf("input: %s\n", argv[1]);
  printf("cities: %d\n", cities);
  printf("population: %d\n", popsize);
  printf("generations: %d\n", generations);

  // start time
  timeval start, end;
  gettimeofday(&start, NULL);

  int besttour[cities];
  const int shortest = tsp(cities, popsize, generations, posx, posy, besttour);

  // end time
  gettimeofday(&end, NULL);
  const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
  printf("compute time: %.3f s\n", runtime);

  // print result
  printf("tour length: %d\n", shortest);

  // draw scaled final tour
  drawTour(cities, posx, posy, besttour);

  return 0;
}

