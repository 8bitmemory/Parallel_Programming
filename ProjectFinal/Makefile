MPI=-O3 -xhost -openmp

all: fractal_hyb2 C2fractal.o CUfractal.o
	@echo done

fractal_hyb2: C2fractal.o CUfractal.o
	mpicxx $(MPI) C2fractal.o CUfractal.o -lcudart -L$$TACC_CUDA_LIB -o fractal_hyb2

C2fractal.o: fractal_hyb2.cpp cs43805351.h
	mpicxx $(MPI) -c fractal_hyb2.cpp -o C2fractal.o 

CUfractal.o: fractal_hyb.cu
	nvcc -O3 -arch=sm_35 -c fractal_hyb.cu -o CUfractal.o

clean: 
	rm *.o 
