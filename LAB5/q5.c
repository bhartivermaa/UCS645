// q5_perfect.c
#include <mpi.h>
#include <stdio.h>

int isPerfect(int n){
    int sum = 1;
    for(int i=2;i<=n/2;i++)
        if(n%i==0) sum += i;
    return sum == n;
}

int main(int argc, char *argv[]){
    int rank, size;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int max = 10000;

    if(rank == 0){
        int num = 2, active = size-1;

        while(active){
            int result;
            MPI_Status status;
            MPI_Recv(&result,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);

            if(num <= max){
                MPI_Send(&num,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
                num++;
            } else {
                int stop = -1;
                MPI_Send(&stop,1,MPI_INT,status.MPI_SOURCE,0,MPI_COMM_WORLD);
                active--;
            }

            if(result > 0) printf("Perfect: %d\n", result);
        }
    } else {
        int req = 0;
        MPI_Send(&req,1,MPI_INT,0,0,MPI_COMM_WORLD);

        while(1){
            int num;
            MPI_Recv(&num,1,MPI_INT,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

            if(num < 0) break;

            int result = isPerfect(num) ? num : -num;
            MPI_Send(&result,1,MPI_INT,0,0,MPI_COMM_WORLD);
        }
    }

    MPI_Finalize();
}
