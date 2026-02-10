#include <iostream>
#include <omp.h>

using namespace std;

int main() {
    long long N = 10000000; 
    double sum_critical = 0.0;
    double sum_reduction = 0.0;
    double start, end;

    // CRITICAL SECTION
    start = omp_get_wtime();
    #pragma omp parallel for
    for (long long i = 0; i < N; i++) {
        #pragma omp critical
        {
            sum_critical += 1.0;
        }
    }
    end = omp_get_wtime();
    cout << "Critical section time: " << (end - start) << " seconds" << endl;

    // REDUCTION
    start = omp_get_wtime();
    #pragma omp parallel for reduction(+:sum_reduction)
    for (long long i = 0; i < N; i++) {
        sum_reduction += 1.0;
    }
    end = omp_get_wtime();
    cout << "Reduction time: " << (end - start) << " seconds" << endl;

    return 0;
}
