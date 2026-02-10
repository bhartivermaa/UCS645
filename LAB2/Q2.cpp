#include <iostream>
#include <omp.h>
#include <chrono>
#include <iomanip>
using namespace std;
using namespace std::chrono;

double compute_pi(long long steps, int threads) {
    double step = 1.0 / steps;
    double sum = 0.0;

    auto start = high_resolution_clock::now();
    #pragma omp parallel num_threads(threads)
    {
        double x, local = 0.0;
        #pragma omp for
        for (long long i = 0; i < steps; i++) {
            x = (i + 0.5) * step;
            local += 4.0 / (1.0 + x * x);
        }
        #pragma omp atomic
        sum += local;
    }
    auto end = high_resolution_clock::now();
    return duration<double>(end - start).count();
}

int main() {
    int maxT = omp_get_max_threads();
    long long base = 100000000;

    cout << "--- STRONG SCALING ---\n";
    long long fixed = 500000000;
    double T1 = compute_pi(fixed, 1);
    cout << "1 thread: " << T1 << "s\n";

    for (int t = 2; t <= maxT; t *= 2) {
        double Tp = compute_pi(fixed, t);
        cout << t << " threads: " << Tp 
             << "  Speedup: " << T1/Tp << endl;
    }

    cout << "\n--- WEAK SCALING ---\n";
    for (int t = 1; t <= maxT; t *= 2) {
        double Tp = compute_pi(base * t, t);
        cout << t << " threads: " << Tp << endl;
    }
}
