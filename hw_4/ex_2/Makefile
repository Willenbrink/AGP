main: main.cu
	nvcc -g -arch=sm_61 -g main.cu -o main

valgrind: main
	valgrind --tool=memcheck ./main $(a) $(b)
