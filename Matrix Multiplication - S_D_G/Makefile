CC=gcc
TARGET=main.exe
CFLAGS = -Wall -fopenmp

all:
	$(CC) $(CFLAGS) main.c MatMul_omp.c -o $(TARGET)

clean:
	rm -f *.o $(TARGET)