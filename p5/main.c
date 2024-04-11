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
    int n, numprocs, rank;
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
        }
        
        // n is sent from root process to all processes
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (!n) break;

        pi = piApprox(rank, numprocs, n);

        /* partial pi approximation is sent to the root process and
           added up to the others to obtain the actual pi approximation */
        MPI_Reduce(&pi, &total_pi, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (!rank/*ROOT PROCESS*/) {
            printf("----------------------------------------\n"
                   "Processes: %d\n"
                   "Intervals: %d\n"
                   "Approximation: %.16f\n"
                   "Error: %.16f\n"
                   "----------------------------------------\n",
                    numprocs, n, total_pi, fabs(total_pi - PI25DT));
        }
    }

    MPI_Finalize();

    return 0;
}
