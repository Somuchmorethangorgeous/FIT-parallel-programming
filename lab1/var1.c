#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>

int M_SIZE = 2;

void printVector(double *v){
    for (int i = 0; i < M_SIZE; ++i){
        printf("%lf ", v[i]);
    }
    putchar('\n');
}

void printMatrix(double *A){
    for (int i = 0; i < M_SIZE; ++i) {
        for (int j = 0; j < M_SIZE; ++j) {
            printf("%lf ", A[i*M_SIZE + j]);
        }
        putchar('\n');
    }
}

double norm(const double *v){
    double tmp = 0;
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


void simpleIterationMethod(const double *A, const double *b, double *x, int numRows){
    const double t = 0.01;
    for (int i = 0; i < numRows; ++i){
        double value = 0;
        for (int j = 0; j < M_SIZE; ++j){
            value += A[i*M_SIZE + j] * x[j];
        }
        x[i] -= t * (value - b[i]);
    }
    printf("\n");
}


void solution(double* A, double *b, double *x, int procNums, int rank){
    int *dataForEachProc = (int*)malloc(sizeof(int) * procNums);
    int *shiftForEachProc = (int*)malloc(sizeof (int) * procNums);
    shiftForEachProc[0] = 0;
    dataForEachProc[0] = (M_SIZE / procNums) * M_SIZE;

    int restRows = M_SIZE;
    int numRows = M_SIZE / procNums;
    for (int i = 1; i < procNums; ++i) {
        restRows -= numRows;
        numRows = restRows / (procNums - i);
        dataForEachProc[i] = numRows * M_SIZE;
        shiftForEachProc[i] = dataForEachProc[i-1] + shiftForEachProc[i-1];
    }

    // пока not free
    double *pieceOfMatrix = (double*)malloc(sizeof(double) * dataForEachProc[rank]);
    MPI_Scatterv(A, dataForEachProc, shiftForEachProc, MPI_DOUBLE, pieceOfMatrix, dataForEachProc[rank],
                 MPI_DOUBLE, 0, MPI_COMM_WORLD);

    x = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        x[i] = 0;
    }
    for (int i = 0; i < procNums; ++i){
        dataForEachProc[i] /= M_SIZE;
        shiftForEachProc[i] /= M_SIZE;
    }
    bool flag = false;
    do {
        simpleIterationMethod(pieceOfMatrix, b, x, dataForEachProc[rank]);
        MPI_Gatherv(x, dataForEachProc[rank], MPI_DOUBLE, x, dataForEachProc,
                    shiftForEachProc, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0){
            flag = answerIsGot(A,b,x);
            MPI_Bcast(&flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    } while (!flag);
    if (rank != 0){
        free(b);
        free(x);
    }
    free(pieceOfMatrix);
    free(dataForEachProc);
    free(shiftForEachProc);
}


void initMatrixAndB(double *A, double *b){
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            A[i*M_SIZE + j] = (i == j) ?  2.0 : 1.0;
        }
    }
    for (int j = 0; j < M_SIZE; ++j){
        u[j] = sin((2*M_PI*j) / M_SIZE);
    }
    for (int m = 0; m < M_SIZE; ++m){
        b[m] = 0;
        for (int n = 0; n < M_SIZE; ++n){
            b[m] += A[m*M_SIZE + n] * u[n];
        }
    }
    free(u);
}


int main(int argc, char **argv) {
    double *A, *b, *x;
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    b = (double*)malloc(sizeof(double) * M_SIZE);
    if (rank == 0) {
        A = (double*)malloc(sizeof(double) * M_SIZE * M_SIZE);
        initMatrixAndB(A, b);
        MPI_Bcast(b, M_SIZE, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    solution(A, b, x, size, rank);
    MPI_Finalize();
    printVector(x);
    free(A);
    free(b);
    free(x);
    return 0;
}
