#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include </usr/include/x86_64-linux-gnu/mpi/mpi.h>

#define DEBUG 1
#define N 9

int ms(struct timeval * tv1, struct timeval * tv2) {
    return ((*tv2).tv_usec - (*tv1).tv_usec) + 1000000 * ((*tv2).tv_sec - (*tv1).tv_sec);
}

int main(int argc, char *argv[]) {

    int i, j, rank, numprocs, default_dist, rest;
    float matrix[N][N], vector[N], result[N];
    struct timeval tv1, tv2;
    double work_ms, comm_ms;

    for (i = 0; i < N; i++) {
        vector[i] = i;
        for (j = 0; j < N; j++) {
            matrix[i][j] = i + j;
        }
    }

    // ramificación de procesos
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    
    // variables para el cálculo de distribución de filas 
    default_dist = N/numprocs;
    rest = N%numprocs;

    // arrays con la cantidad y el desplazamiento de filas de cada proceso
    int rows_per_process[numprocs], counts[numprocs], desplaz[numprocs];

    // cálculo de las filas que le corresponden a cada proceso desde la raíz
    if (!rank) {
        for (int i = 0; i < numprocs; i++) {
            rows_per_process[i] = default_dist + (i < rest? 1:0);
            counts[i] = rows_per_process[i]*N;
            desplaz[i] = i > 0? desplaz[i-1] + counts[i-1] : 0;
        }
    }

    // envio de los arrays a todos los procesos
    MPI_Bcast(rows_per_process, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(counts, numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(desplaz, numprocs, MPI_INT, 0, MPI_COMM_WORLD);

    // número de filas correspondientes al proceso actual
    int M = rows_per_process[rank];

    // matriz y resultado propio de cada proceso
    float work_matrix[M][N], work_result[M];

    printf("P%d: %d, %d, %d\n", rank, M, counts[rank], desplaz[rank]);

    gettimeofday(&tv1, NULL); // start

    // se envia la parte de la matriz que corresponde a cada proceso, y el vector entero
    MPI_Scatterv(matrix, counts, desplaz, MPI_FLOAT, work_matrix, M*N, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    gettimeofday(&tv2, NULL); // end
    
    comm_ms = ms(&tv1, &tv2);

    gettimeofday(&tv1, NULL); // start

    // se calcula el resultado
    for(i = 0; i < M; i++) {
        work_result[i] = 0;
        for(j = 0; j < N; j++) {
            work_result[i] += work_matrix[i][j] * vector[j];
        }
    }
    
    gettimeofday(&tv2, NULL); // end

    work_ms = ms(&tv1, &tv2);

    gettimeofday(&tv1, NULL); // start

    MPI_Gatherv(work_result, M, MPI_FLOAT, result + (desplaz[rank]/N), counts, desplaz, MPI_FLOAT, 0, MPI_COMM_WORLD);

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