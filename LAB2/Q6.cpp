#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace std::chrono;

int main() {
    // Size large enough to exceed cache (memory-bound)
    long long N = 30000000;  //30 million elements

    vector<double> A(N, 0.0), B(N, 1.1), C(N, 2.2);
    double alpha = 3.3;

    int max_threads = omp_get_max_threads();

    cout << "Memory Bandwidth Saturation Test (Vector Triad)\n";
    cout << left << setw(10) << "Threads"
         << setw(15) << "Time(s)"
         << setw(15) << "BW(GB/s)"
         << "Speedup\n";
    cout << string(55, '-') << endl;

    double T1 = 0.0;

    for (int threads = 1; threads <= max_threads; threads *= 2) {

        auto start = high_resolution_clock::now();

        #pragma omp parallel for num_threads(threads)
        for (long long i = 0; i < N; i++) {
            A[i] = B[i] + alpha * C[i];
        }

        auto end = high_resolution_clock::now();
        double Tp = duration<double>(end - start).count();

        if (threads == 1)
            T1 = Tp;

        //Read B, Read C, Write A
        double total_bytes = 3.0 * N * sizeof(double);
        double bandwidth = (total_bytes / Tp) / 1e9; // GB/s
        double speedup = T1 / Tp;

        cout << left << setw(10) << threads
             << setw(15) << Tp
             << setw(15) << bandwidth
             << fixed << setprecision(2) << speedup << "x\n";
    }

    return 0;
}

