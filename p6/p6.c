#include <stdio.h>
#include <sys/time.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

#define DEBUG 1
#define N 9

int ms(struct timeval * tv1, struct timeval * tv2) {
    return ((*tv2).tv_usec - (*tv1).tv_usec) + 1000000 * ((*tv2).tv_sec - (*tv1).tv_sec);
}

int main(int argc, char *argv[]) {

    int i, j, rank, numprocs, M, rest;
    float matrix[N][N], vector[N], result[N];
    struct timeval tv1, tv2;
    double work_ms, comm_ms;

    for (i = 0; i < N; i++) {
        vector[i] = i;
        for (j = 0; j < N; j++) {
            matrix[i][j] = i + j;
        }
    }

    // process ramification
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    M = N/numprocs;
    rest = N%numprocs;

    int desplaz[numprocs], count[numprocs];
    if (rank == 0) {
        for (int i = 0; i < numprocs; i++) {
            count[i] = i < rest? M + 1 : M;
            desplaz[i] = i > 0? desplaz[i-1] + count[i-1] : 0;
        }
    }

    

    float work_matrix[count[rank]][N], work_result[count[rank]];

    printf("P%d: %d\n", rank, M);

    gettimeofday(&tv1, NULL); // start

    // se envia la parte de la matriz que corresponde a cada proceso
    MPI_Scatterv(matrix, count, desplaz, MPI_FLOAT, work_matrix, M*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv2, NULL); // end
    
    comm_ms = ms(&tv1, &tv2);

    gettimeofday(&tv1, NULL); // start

    for(i = 0; i < M; i++) {
        work_result[i] = 0;
        for(j = 0; j < N; j++) {
            work_result[i] += work_matrix[i][j] * vector[j];
        }
    }
    
    gettimeofday(&tv2, NULL); // end

    work_ms = ms(&tv1, &tv2);

    gettimeofday(&tv1, NULL); // start

    MPI_Gather(work_result, M, MPI_FLOAT, result, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv2, NULL); // end

    comm_ms += ms(&tv1, &tv2);

    if (!rank) {
        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf("%f\t ", result[i]);
            }
            printf("\n");
        } else {
            printf("Work time (seconds) = %lf\n", (double) work_ms/1E6);
            printf("Comm time (seconds) = %lf\n", (double) comm_ms/1E6);
        } 
    }   

    MPI_Finalize();

    return 0;
}