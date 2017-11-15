
all: ADMM.o  test.o
ADMM.o: main.c
	mpicc main.c -lm -lrt
test.o: test.c
	gcc test.c -o test.o
clean:
	rm *.out
