#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;

#define CACHE_LINE 64

// BAD causes false sharing
struct Bad {
    long long value;
};

// GOOD padding avoids false sharing
struct Good {
    long long value;
    char padding[CACHE_LINE - sizeof(long long)];
};

int main() {
    int threads = omp_get_max_threads();
    long long iterations = 100000000;

    vector<Bad> bad(threads);
    vector<Good> good(threads);

    double start, end;

    // FALSE SHARING VERSION
    start = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long long i = 0; i < iterations; i++) {
            bad[id].value++;
        }
    }
    end = omp_get_wtime();
    cout << "False sharing time: " << (end - start) << " seconds" << endl;

    // PADDED VERSION
    start = omp_get_wtime();
    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        for (long long i = 0; i < iterations; i++) {
            good[id].value++;
        }
    }
    end = omp_get_wtime();
    cout << "Padded version time: " << (end - start) << " seconds" << endl;

    return 0;
}
