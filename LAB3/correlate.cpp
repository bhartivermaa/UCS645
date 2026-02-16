#include <cmath>

void correlate(int ny, int nx, const float* data, float* result)
{
    for(int i = 0; i < ny; i++)
    {
        for(int j = 0; j <= i; j++)
        {
            double sum_i = 0, sum_j = 0;

            for(int k = 0; k < nx; k++)
            {
                sum_i += data[k + i*nx];
                sum_j += data[k + j*nx];
            }

            double mean_i = sum_i / nx;
            double mean_j = sum_j / nx;

            double num = 0, denom_i = 0, denom_j = 0;

            for(int k = 0; k < nx; k++)
            {
                double xi = data[k + i*nx] - mean_i;
                double xj = data[k + j*nx] - mean_j;

                num += xi * xj;
                denom_i += xi * xi;
                denom_j += xj * xj;
            }

            result[i + j*ny] =
            num / sqrt(denom_i * denom_j);
        }
    }
}
