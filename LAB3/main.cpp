#include <iostream>
#include <cstdlib>
#include <chrono>

using namespace std;

void correlate(int ny, int nx, const float* data, float* result);

int main(int argc, char* argv[])
{
    if(argc < 3)
    {
        cout << "Usage: ./corr ny nx\n";
        return 0;
    }

    int ny = atoi(argv[1]);
    int nx = atoi(argv[2]);

    float* data = new float[ny * nx];
    float* result = new float[ny * ny];

    for(int i = 0; i < ny * nx; i++)
        data[i] = rand() % 10;

    auto start = chrono::high_resolution_clock::now();

    correlate(ny, nx, data, result);

    auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> time = end - start;

    cout << "Execution Time: " << time.count() << " seconds\n";

    delete[] data;
    delete[] result;

    return 0;
}
