#include <stdio.h>
#include <math.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

double piApprox(int rank, int numprocs, int n) {
    double h, sum, x;
    int i;

    h = 1. / (double) n;
    sum = 0.;
    for (i = rank; i < n; i += numprocs) {
        x = h * ((double)i - .5);
        sum += 4. / (1. + x*x);
    }
    return h * sum;
}


int main(int argc, char *argv[]) {
    int i, n, numprocs, rank;
    const double PI25DT = 3.141592653589793238462643;
    double pi, total_pi = 0;

    // process ramification
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    while (1) {

        if (!rank/*ROOT PROCESS*/) {
            printf("Enter the number of intervals (0 quits): \n");
            scanf("%d", &n);
        
            for (i = 0; i < numprocs; i++) { 
                // n is sent from root process to all processes
                MPI_Send(&n, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            }
        }
        
        // n is received from root process
        MPI_Recv(&n, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (!n) break;

        pi = piApprox(rank, numprocs, n);

        // partial pi approximation is sent to the root process
        MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD); 

        if (!rank/*ROOT PROCESS*/) {
            for (i = 0; i < numprocs; i++) { 
                // pi partial approximations are received in the root process
                MPI_Recv(&pi, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE); 
                // real approximation is obtained by adding up every partial approximation
                total_pi += pi; 
            }

            printf("----------------------------------------\n"
                   "Processes: %d\n"
                   "Intervals: %d\n"
                   "Approximation: %.16f\n"
                   "Error: %.16f\n"
                   "----------------------------------------\n",
                    numprocs, n, total_pi, fabs(total_pi - PI25DT));
            total_pi = 0;
        }
    }

    MPI_Finalize();

    return 0;
}
