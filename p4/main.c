#include <stdio.h>
#include <math.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

int main(int argc, char *argv[]) {

    int i, n, numprocs, rank;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;
    
    while (1) {
        printf("Enter the number of intervals: (0 quits) \n");
        scanf("%d",&n);
        if (!n) break;

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

        MPI_Send(&pi, 1, MPI_DOUBLE, rank, MPI_ANY_TAG, MPI_COMM_WORLD);
        MPI_Recv(&pi, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  
        h   = 1.0 / (double) n;
        sum = 0.0;
        for (i = 1; i <= n; i+=numprocs) {
            x = h * ((double)i - 0.5);
            sum += 4.0 / (1.0 + x*x);
        }
        pi = h * sum;

        //MPI_Send(&pi, 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD);
        //MPI_Recv(&pi, 1, MPI_DOUBLE, rank, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("pi is approximately %.16f, Error is %.16f\n", pi, fabs(pi - PI25DT));

        MPI_Finalize();
    }
}
