#include <iostream>
#include <vector>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {
    //safe size for WSL (still memory-bound)
    long long N = 30000000;  // 30 million elements

    vector<double> A(N, 0.0), B(N, 1.0), C(N, 2.0);
    double alpha = 0.5;

    int threads = omp_get_max_threads();

    auto start = high_resolution_clock::now();

    #pragma omp parallel for num_threads(threads)
    for (long long i = 0; i < N; i++) {
        A[i] = B[i] + alpha * C[i];
    }

    auto end = high_resolution_clock::now();
    duration<double> elapsed = end - start;

    // Total data moved: Read B + Read C + Write A = 3 × N × 8 bytes
    double total_bytes = 3.0 * N * sizeof(double);
    double bandwidth_gb_s = (total_bytes / elapsed.count()) / 1e9;

    cout << "Threads   : " << threads << endl;
    cout << "Time      : " << elapsed.count() << " s" << endl;
    cout << "Bandwidth : " << bandwidth_gb_s << " GB/s" << endl;

    return 0;
}
