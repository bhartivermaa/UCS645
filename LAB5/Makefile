CC = mpicc
CFLAGS = -O2

all: q1 q2 q3 q4 q5

q1:
	$(CC) $(CFLAGS) q1_daxpy.c -o q1

q2:
	$(CC) $(CFLAGS) q2_broadcast.c -o q2

q3:
	$(CC) $(CFLAGS) q3_dotproduct.c -o q3

q4:
	$(CC) $(CFLAGS) q4_primes.c -o q4 -lm

q5:
	$(CC) $(CFLAGS) q5_perfect.c -o q5

clean:
	rm -f q1 q2 q3 q4 q5
