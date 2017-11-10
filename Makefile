
all: ADMM.o
ADMM.o: main.c
	mpicc main.c -lm
clean:
	rm *.out
