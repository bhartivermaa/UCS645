#include <omp.h>
#include <iostream>
#include <vector>

#define N 500

int main() {

    std::vector<std::vector<double>> grid(N, std::vector<double>(N,0));

    double start = omp_get_wtime();

    for(int t=0; t<100; t++) {

        #pragma omp parallel for collapse(2)
        for(int i=1; i<N-1; i++)
            for(int j=1; j<N-1; j++)
                grid[i][j] = (grid[i-1][j] + grid[i+1][j] +
                              grid[i][j-1] + grid[i][j+1]) / 4;
    }

    double end = omp_get_wtime();

    std::cout << "Time: " << end-start << " seconds\n";

    return 0;
}
