#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <malloc.h>
#include <mpi.h>

const int n1 = 5;
const int n2 = 8;
const int n3 = 5;


double* initMatrix(const int X_dim, const int Y_dim){
    double *M = (double*)malloc(sizeof(double) * X_dim * Y_dim);
    for (int i = 0; i < X_dim; ++i){
        for (int j = 0; j < Y_dim; ++j){
            M[i*Y_dim + j] = rand()%10;
            printf("%lf ",  M[i*Y_dim + j]);
        } putchar('\n');
    }
    return M;
}

double* dataDistributionA(double *A, int *dataForEachProc, int *shiftForEachProc, int numProcs, int rank){
    dataForEachProc[0] = (n1 / numProcs) * n2;
    shiftForEachProc[0] = 0;
    int restRows = n1;
    for (int i = 1; i < numProcs; ++i){
        restRows -= dataForEachProc[i-1] / n2;
        dataForEachProc[i] = restRows * n2 / (numProcs - i);
        shiftForEachProc[i] = shiftForEachProc[i-1] + dataForEachProc[i-1];
    }
    double *blockData = (double*)malloc(sizeof(double) * dataForEachProc[rank]);
    MPI_Scatterv(A, dataForEachProc, shiftForEachProc, MPI_DOUBLE, blockData, dataForEachProc[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return blockData;
}

int main(int argc, char **argv) {
    double *A, *B, *C;
    double *blockDataA, *blockDataB;
    int *dataForEachProc, *shiftForEachProc;
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dataForEachProc = (int*)malloc(sizeof(int) * size);
    shiftForEachProc = (int*)malloc(sizeof(int) * size);

    if (rank == 0){
        srand(time(NULL));
        A = initMatrix(n1, n2);
        B = initMatrix(n2, n3);
        C = (double*)malloc(sizeof(double) * n1 * n3);
    }

    blockDataA = dataDistributionA(A, dataForEachProc, shiftForEachProc, size, rank);

    if (rank == 2){
        putchar('\n');
        for (int i = 0; i < dataForEachProc[rank] / n2; ++i){
            for (int j = 0; j < n2; ++j){
                printf("%lf ", blockDataA[i*n2 + j]);
            } putchar('\n');
        }
    }

    MPI_Finalize();
    if (rank == 0){
        free(A);
        free(B);
        free(C);
    }

    free(blockDataA);
    free(dataForEachProc);
    free(shiftForEachProc);
    return 0;
}
