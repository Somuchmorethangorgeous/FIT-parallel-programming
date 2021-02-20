#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>

int M_SIZE = 10;


void printVector(const double *v){
    for (int i = 0; i < M_SIZE; ++i){
        printf("%lf ", v[i]);
    }
    putchar('\n');
}


double norm(const double *v){
    double tmp = 0.0;
    for (int i = 0; i < M_SIZE; ++i){
        tmp += v[i] * v[i];
    }
    return sqrt(tmp);
}


bool answerIsGot(const double *A, const double *b, const double *x){
    const double e = pow(10, -6);
    double *sol = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i) {
        double value = 0;
        for (int j = 0; j < M_SIZE; ++j) {
            value += A[i * M_SIZE + j] * x[j];
        }
        sol[i] = value - b[i];
    }
    if (norm(sol) / norm (b) < e){
        free(sol);
        return true;
    } else {
        free(sol);
        return false;
    }
}


void simpleIterationMethod(const double *A, const double *b, double *x, int shift, int numRows){
    const double t = 0.01;
    for (int i = 0; i < numRows; ++i){
        double value = 0.0;
        for (int j = 0; j < M_SIZE; ++j){
            value += A[i*M_SIZE + j] * x[j];
        }
        x[i + shift] -= t * (value - b[i+shift]);
    }
}


double* solution(double* A, double *b, double *blockData, int numProcs, int rank){
    double *x = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        x[i] = 0;
    }

    int *updCoord = (int*)malloc(sizeof (int) * M_SIZE);
    int *shiftCoord = (int*)malloc(sizeof (int) * M_SIZE);
    updCoord[0] = M_SIZE / numProcs;
    shiftCoord[0] = 0;

    int restRows = M_SIZE;
    for (int i = 1; i < numProcs; ++i){
        restRows -= updCoord[i-1];
        updCoord[i] = restRows / (numProcs - i);
        shiftCoord[i] = shiftCoord[i-1] + updCoord[i-1];
    }

    bool flag = false;
    do {
        simpleIterationMethod(blockData, b, x, shiftCoord[rank], updCoord[rank]);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x, updCoord,
                       shiftCoord, MPI_DOUBLE, MPI_COMM_WORLD);
        if (rank == 0){
            flag = answerIsGot(A,b,x);
        }
        MPI_Bcast(&flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    } while (!flag);
    free(updCoord);
    free(shiftCoord);
    return x;
}


double* cutMatrix(double *A, int numProcs, int rank){
    int *dataForEachProc = (int*)malloc(sizeof(int) * numProcs);
    int *shiftForEachProc = (int*)malloc(sizeof(int) * numProcs);
    shiftForEachProc[0] = 0;
    dataForEachProc[0] = (M_SIZE / numProcs) * M_SIZE;

    int restRows = M_SIZE;
    int numRows = M_SIZE / numProcs;
    for (int i = 1; i < numProcs; ++i) {
        restRows -= numRows;
        numRows = restRows / (numProcs - i);
        dataForEachProc[i] = numRows * M_SIZE;
        shiftForEachProc[i] = dataForEachProc[i-1] + shiftForEachProc[i-1];
    }

    double *blockData = (double*)malloc(sizeof(double) * dataForEachProc[rank]);
    MPI_Scatterv(A, dataForEachProc, shiftForEachProc, MPI_DOUBLE, blockData, dataForEachProc[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);
    free(dataForEachProc);
    free(shiftForEachProc);
    return blockData;
}



void initMatrixAndB(double *A, double *b){
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            A[i*M_SIZE+j] = (i == j) ?  2.0 : 1.0;
        }
        u[i] = sin((2*M_PI*i) / M_SIZE);
    }
    for (int m = 0; m < M_SIZE; ++m){
        b[m] = 0;
        for (int n = 0; n < M_SIZE; ++n){
            b[m] += A[m*M_SIZE + n] * u[n];
        }
    }
    printf("Answer is: ");
    printVector(u);
    putchar('\n');
    free(u);
}


int main(int argc, char **argv) {
    double *A, *b, *x, *blockData;
    int size, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    b = (double*)malloc(sizeof(double) * M_SIZE);
    if (rank == 0) {
        A = (double*)malloc(sizeof(double) * M_SIZE * M_SIZE);
        initMatrixAndB(A, b);
    }

    MPI_Bcast(b, M_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    blockData = cutMatrix(A, size, rank);
    x = solution(A, b, blockData, size, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0) {
        free(A);
    }
    printf("Solution of #%d process is: ", rank);
    printVector(x);
    free(blockData);
    free(b);
    free(x);
    return 0;
}
