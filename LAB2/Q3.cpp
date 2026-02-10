#include <iostream>
#include <omp.h>
#include <cmath>

using namespace std;

void heavy_work(int i) {
    double dummy = 0.0;
    for (int j = 0; j < i * 10000; j++) {
        dummy += sin(j) * cos(j);
    }
}

int main() {
    int N = 1000;
    double start, end;

    // STATIC
    start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    cout << "Static scheduling time: " << (end - start) << " seconds" << endl;

    // DYNAMIC
    start = omp_get_wtime();
    #pragma omp parallel for schedule(dynamic, 10)
    for (int i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    cout << "Dynamic scheduling time: " << (end - start) << " seconds" << endl;

    // GUIDED
    start = omp_get_wtime();
    #pragma omp parallel for schedule(guided)
    for (int i = 0; i < N; i++) {
        heavy_work(i);
    }
    end = omp_get_wtime();
    cout << "Guided scheduling time: " << (end - start) << " seconds" << endl;

    return 0;
}

