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
        }
    }
    return M;
}


double* calculateMatrixC(const int X_dim, const int Y_dim, const double *blockDataA, const double *blockDataB){
    double *blockDataC = (double*)malloc(sizeof(double) * X_dim * Y_dim);
    for (int i = 0; i < X_dim; ++i){
        for (int j = 0; j < Y_dim; ++j){
            blockDataC[i*Y_dim + j] = 0.0;
            for (int k = 0; k < n2; ++k){
                blockDataC[i*Y_dim + j] += blockDataA[i*n2 + k] * blockDataB[j*n2 + k];
            }
        }
    }
    return blockDataC;
}


double *dataDistributionB(double* B, int *dataForEachProc, int *shiftForEachProc, int numProcs, int rank){
    MPI_Datatype allCol, coltype;

    MPI_Type_vector(n2, 1, n3, MPI_DOUBLE, &allCol);
    MPI_Type_commit(&allCol);
    MPI_Type_create_resized(allCol, 0, sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    dataForEachProc[0] = (n3 / numProcs);
    shiftForEachProc[0] = 0;
    int restCols = n3;
    for (int i = 1; i < numProcs; ++i){
        restCols -= dataForEachProc[i-1];
        dataForEachProc[i] = restCols  / (numProcs - i);
        shiftForEachProc[i] = shiftForEachProc[i-1] + dataForEachProc[i-1];
    }

    double *blockData = (double*)malloc(sizeof(double) * dataForEachProc[rank] * n2);

    MPI_Scatterv(B, dataForEachProc, shiftForEachProc, coltype, blockData, dataForEachProc[rank] * n2,
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    return blockData;
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
    double *blockDataA, *blockDataB, *blockDataC;
    int *dataForEachProc, *shiftForEachProc;
    int x_C, y_C;
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
    x_C = dataForEachProc[rank] / n2;
    if (rank == 0){
        for (int i = 0; i < x_C; ++i){
            for (int j = 0; j < n2; ++j){
                printf("%lf ", blockDataA[i*n2 + j]);
            } putchar('\n');
        }putchar('\n');
    }

    blockDataB = dataDistributionB(B, dataForEachProc, shiftForEachProc, size, rank);
    y_C = dataForEachProc[rank];

    if (rank == 0) {
        for (int i = 0; i < dataForEachProc[rank]; ++i) {
            for (int j = 0; j < n2; ++j) {
                printf("%lf ", blockDataB[i * n2 + j]);
            } putchar('\n');
        }
        putchar('\n');
        blockDataC = calculateMatrixC(x_C, y_C, blockDataA, blockDataB);
       for (int i = 0; i < x_C; ++i){
            for (int j = 0; j < y_C; ++j){
                printf("%lf ", blockDataC[i*y_C + j]);
            } putchar('\n');
        }
    }

    MPI_Finalize();
    if (rank == 0){
        free(A);
        free(B);
        free(C);
        free(blockDataC);
    }

    free(blockDataA);
    free(blockDataB);
    free(dataForEachProc);
    free(shiftForEachProc);
    return 0;
}
