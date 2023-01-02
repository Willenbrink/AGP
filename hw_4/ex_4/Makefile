# -allow-unsupported-compiler is necessary on my distribution for cuda-memcheck
# (which is not available in cuda-12 for some reason)
# Running cuda-12 does not require this and it does not affect the error produced.
main: main.cu
	nvcc -g -G -allow-unsupported-compiler -lcublas -lcusparse -arch=sm_61 -g main.cu -o main

valgrind: main
	valgrind --tool=memcheck ./main $(a) $(b)
