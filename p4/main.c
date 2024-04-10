#include <stdio.h>
#include <math.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

int main(int argc, char *argv[]) {
    int i, n, numprocs, rank;
    const double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x, total_pi = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    while (1) {
        if (!rank) {
            printf("Enter the number of intervals (0 quits): \n");
            scanf("%d", &n);
        
            for (i = 0; i < numprocs; i++) { // n is sent from root process to all processes
                MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // n is received from root process
        if (!n) break;

        h   = 1.0 / (double) n;
        sum = 0.0;
        for (i = rank; i < n; i += numprocs) {
            x = h * ((double)i - 0.5);
            sum += 4.0 / (1.0 + x*x);
        }
        pi = h * sum;

        MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); // partial pi approximation is sent to the root process

        if (!rank) {
            for (i = 0; i < numprocs; i++) { 
                MPI_Recv(&pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); // pi partial approximations are received
                total_pi += pi; // real approximation is obtained by adding up every partial approximation
            }

            printf("PI is approximately %.16f, with an error of %.16f\n", total_pi, fabs(total_pi - PI25DT));
            total_pi = 0;
        }
    }

    MPI_Finalize();

    return 0;
}
