#include <omp.h>
#include <iostream>
#include <vector>
using namespace std;

int main(){
	long N = 100000000;
	vector<double> A(N, 1.0), B(N,1.0), C(N,0.0);
	double start = omp_get_wtime();
	#pragma omp parallel for
	for (long i=0; i<N; i++)
		C[i] = A[i] + B[i];

	double end = omp_get_wtime();
	cout<<"Time = "<<(end-start) << " seconds\n";
}
