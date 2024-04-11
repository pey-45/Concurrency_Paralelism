#include <stdio.h>
#include <sys/time.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

#define DEBUG 0
#define N 5

int main(int argc, char *argv[]) {

    int i, j, rank, numprocs;
    float matrix[N][N], result[N], vector[N];
    struct timeval tv1, tv2;

    // process ramification
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    const int M = N/numprocs + 1; // +1 to avoid unasigned rows
    float work_matrix[M][N], work_result[M];

    if (!rank/*ROOT*/) {
        // matrix and vector initialization
        for (i = 0; i < N; i++) {
            vector[i] = i;
            for (j = 0; j < N; j++) {
                matrix[i][j] = i + j;
            }
        }
    }

    // se envia la parte de la matriz que corresponde a cada proceso
    MPI_Scatter(matrix, M*N, MPI_FLOAT, work_matrix, M*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    gettimeofday(&tv1, NULL);

    for(i = 0; i < M; i++) {
        work_result[i] = 0;
        for(j = 0; j < N; j++) {
            work_result[i] += work_matrix[i][j] * vector[j];
        }
    }

    MPI_Gather(work_result, M, MPI_FLOAT, result, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv2, NULL);
        
    int microseconds = (tv2.tv_usec - tv1.tv_usec) + 1000000 * (tv2.tv_sec - tv1.tv_sec);

    if (!rank) {
        if (DEBUG) {
            for (i = 0; i < N; i++) {
                printf("%f\t ", result[i]);
            }
            printf("\n");
        } else {
            printf("Time (seconds) = %lf\n", (double) microseconds/1E6);
        } 
    }   

    MPI_Finalize();

    return 0;
}