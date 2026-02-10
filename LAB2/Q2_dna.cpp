#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>

#define N 1000

int main() {

    // Create a DP scoring matrix
    std::vector<std::vector<int>> dp(N, std::vector<int>(N, 0));

    double start = omp_get_wtime();

    // Wavefront (diagonal) parallelization
    for(int k = 0; k < N; k++) {

        #pragma omp parallel for schedule(dynamic)
        for(int i = 0; i <= k; i++) {

            int j = k - i;

            if(i > 0 && j > 0) {
                int match = dp[i-1][j-1] + 1;
                int deleteGap = dp[i-1][j];
                int insertGap = dp[i][j-1];

                dp[i][j] = std::max({0, match, deleteGap, insertGap});
            }
        }
    }

    double end = omp_get_wtime();

    std::cout << "Time: " << end - start << " seconds\n";

    return 0;
}
