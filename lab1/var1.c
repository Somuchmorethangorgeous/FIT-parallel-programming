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


bool answerIsGot(const double *blockData, const double *b, const double *x, const int *rowsForEachProc, const int *shiftVec, int rank, const double normB){
    int shift = shiftVec[rank];
    const double e = pow(10, -6);
    double *sol = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < rowsForEachProc[rank]; ++i) {
        double value = 0;
        for (int j = 0; j < M_SIZE; ++j) {
            value += blockData[i * M_SIZE + j] * x[j];
        }
        sol[i+shift] = value - b[i+shift];
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, sol, rowsForEachProc, shiftVec, MPI_DOUBLE, MPI_COMM_WORLD);

    bool result = norm(sol) / normB < e;
    free(sol);
    return result;
}




void simpleIterationMethod(const double *blockData, const double *b, double *x, int numRows, int shift){
    const double t = 0.01;
    for (int i = 0; i < numRows; ++i){
        double value = 0.0;
        for (int j = 0; j < M_SIZE; ++j){
            value += blockData[i*M_SIZE + j] * x[j];
        }
        x[i + shift] -= t * (value - b[i+shift]);
    }
}


double* solution(const double *blockData, const double *b, int *dataVec, int *shiftVec, int rank, const double normB){
    double *x = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        x[i] = 0.0;
    }
    bool isFinish = false;
    do {
        simpleIterationMethod(blockData, b, x, dataVec[rank], shiftVec[rank]);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, x, dataVec,
                       shiftVec, MPI_DOUBLE, MPI_COMM_WORLD);
        isFinish = answerIsGot(blockData, b, x, dataVec, shiftVec, rank, normB);
    } while (!isFinish);
    return x;
}


void initMatrixAndB(double *blockData, double *b, const int* rowsForEachProc, const int *shiftForEachProc, int rank){
    int shift = shiftForEachProc[rank];
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < rowsForEachProc[rank]; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            blockData[i*M_SIZE+j] = (i + shift == j) ?  2.0 : 1.0;
        }
        u[i+shift] = sin((2*M_PI*(i+shift) / M_SIZE));
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, u, rowsForEachProc, shiftForEachProc, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < rowsForEachProc[rank]; ++i){
        b[i+shift] = 0;
        for (int j = 0; j < M_SIZE; ++j){
            b[i+shift] += blockData[i*M_SIZE + j] * u[j];
        }
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, b, rowsForEachProc,
                   shiftForEachProc, MPI_DOUBLE, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Answer is: ");
        printVector(u);
        putchar('\n');
    }
    free(u);
}


double* cutMatrix(double *b, int *dataVec, int *shiftVec, int rank){
    double *blockData = (double*)malloc(sizeof(double) * dataVec[rank] * M_SIZE);
    initMatrixAndB(blockData, b, dataVec, shiftVec, rank);
    return blockData;
}


 void dataDistribution(int *dataVec, int*shiftVec, int numProcs){
     dataVec[0] = M_SIZE / numProcs;
     shiftVec[0] = 0;
     int restRows = M_SIZE;
     for (int i = 1; i < numProcs; ++i){
         restRows -= dataVec[i-1];
         dataVec[i] = restRows / (numProcs - i);
         shiftVec[i] = shiftVec[i-1] + dataVec[i-1];
     }
}


int main(int argc, char **argv) {
    double *b, *x, *blockData;
    int *dataVec, *shiftVec;
    int size, rank;
    double normB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dataVec = (int*)malloc(sizeof(int) * size);
    shiftVec = (int*)malloc(sizeof(int) * size);
    dataDistribution(dataVec, shiftVec, size);

    b = (double*)malloc(sizeof(double) * M_SIZE);
    blockData = cutMatrix(b, dataVec, shiftVec, rank);
    normB = norm(b);

    x = solution(blockData, b, dataVec, shiftVec, rank, normB);

    MPI_Finalize();

    printf("Solution of #%d process is: ", rank);
    printVector(x);
    free(blockData);
    free(dataVec);
    free(shiftVec);
    free(b);
    free(x);
    return 0;
}
