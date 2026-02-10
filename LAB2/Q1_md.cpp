#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>

#define NUM_PARTICLES 1000

int main() {

    std::vector<double> posX(NUM_PARTICLES),
                        posY(NUM_PARTICLES),
                        posZ(NUM_PARTICLES);

    std::vector<double> forceX(NUM_PARTICLES, 0.0),
                        forceY(NUM_PARTICLES, 0.0),
                        forceZ(NUM_PARTICLES, 0.0);

    // Initialize particle positions
    for(int p = 0; p < NUM_PARTICLES; p++) {
        posX[p] = rand() % 100;
        posY[p] = rand() % 100;
        posZ[p] = rand() % 100;
    }

    double t_start = omp_get_wtime();

    #pragma omp parallel for
    for(int p = 0; p < NUM_PARTICLES; p++) {

        double fx_local = 0.0;
        double fy_local = 0.0;
        double fz_local = 0.0;

        for(int q = 0; q < NUM_PARTICLES; q++) {
            if(p == q) continue;

            double dx = posX[p] - posX[q];
            double dy = posY[p] - posY[q];
            double dz = posZ[p] - posZ[q];

            double distance = sqrt(dx*dx + dy*dy + dz*dz) + 0.0001;

            double force_mag = 1.0 / (distance * distance);

            fx_local += force_mag * dx;
            fy_local += force_mag * dy;
            fz_local += force_mag * dz;
        }

        forceX[p] = fx_local;
        forceY[p] = fy_local;
        forceZ[p] = fz_local;
    }

    double t_end = omp_get_wtime();

    std::cout << "Execution Time: " << (t_end - t_start) << " seconds\n";

    return 0;
}

