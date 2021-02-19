#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>

int M_SIZE = 16;


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


void simpleIterationMethod(const double *A, const double *b, double *x, const int* numCols, int rank){
    double* mulVec = (double*)malloc(sizeof(double) * M_SIZE);
    double* resVec = (double*)malloc(sizeof(double) * numCols[rank]);
    const double t = 0.01;
    for (int i = 0; i < M_SIZE; ++i){
        mulVec[i] = 0;
    }
    for (int i = 0; i < numCols[rank]; ++i) {
        for (int j = 0; j < M_SIZE; ++j) {
            mulVec[j] += A[i*M_SIZE+j] * x[i];
        }
    }

    MPI_Reduce_scatter(mulVec, resVec, numCols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < numCols[rank]; ++i){
        x[i] -= t * (resVec[i] - b[i]);
    }

    free(mulVec);
    free(resVec);
}


void solution(double* A, double *b, double *x, double *blockData, int numProcs, int rank){
    if (rank == 0) {
        for (int i = 0; i < M_SIZE; ++i) {
            x[i] = 0;
        }
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

    double *partVecX = (double*)malloc(sizeof(double) * updCoord[rank]);
    double *partVecB = (double*)malloc(sizeof(double) * updCoord[rank]);

    MPI_Scatterv(x, updCoord, shiftCoord, MPI_DOUBLE, partVecX, updCoord[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(b, updCoord, shiftCoord, MPI_DOUBLE, partVecB, updCoord[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    bool flag = false;
    do {
        simpleIterationMethod(blockData, partVecB, partVecX, updCoord, rank);
        MPI_Gatherv(partVecX, updCoord[rank], MPI_DOUBLE, x, updCoord,
                       shiftCoord, MPI_DOUBLE,0, MPI_COMM_WORLD);
        if (rank == 0){
            flag = answerIsGot(A,b,x);
        }
        MPI_Bcast(&flag, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    } while (!flag);

    free(partVecX);
    free(partVecB);
    free(updCoord);
    free(shiftCoord);
}


double* cutMatrix(double *A, int numProcs, int rank){
    MPI_Datatype allCol, coltype;

    MPI_Type_vector(M_SIZE, 1, M_SIZE, MPI_DOUBLE, &allCol);
    MPI_Type_commit(&allCol);
    MPI_Type_create_resized(allCol, 0, sizeof(double), &coltype);
    MPI_Type_commit(&coltype);

    int* dataForEachProc = (int*)malloc(sizeof(int) * numProcs);
    int *shiftForEachProc = (int*)malloc(sizeof(int) * numProcs);

    dataForEachProc[0] = (M_SIZE / numProcs);
    shiftForEachProc[0] = 0;

    int restCols = M_SIZE;
    int numCols = M_SIZE / numProcs;
    for (int i = 1; i < numProcs; ++i) {
        restCols -= numCols;
        numCols = restCols / (numProcs - i);
        dataForEachProc[i] = numCols;
        shiftForEachProc[i] = dataForEachProc[i-1] + shiftForEachProc[i-1];
    }

    double *blockData = (double*)malloc(sizeof(double) * dataForEachProc[rank] * M_SIZE);

    MPI_Scatterv(A, dataForEachProc, shiftForEachProc, coltype, blockData, dataForEachProc[rank] * M_SIZE,
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

    if (rank == 0) {
        x = (double*)malloc(sizeof(double) * M_SIZE);
        b = (double*)malloc(sizeof(double) * M_SIZE);
        A = (double*)malloc(sizeof(double) * M_SIZE * M_SIZE);
        initMatrixAndB(A, b);
    }

    blockData = cutMatrix(A, size, rank);
    solution(A, b, x, blockData, size, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    if (rank == 0) {
        printf("Answer on root process is: ");
        printVector(x);
        free(b);
        free(x);
        free(A);
    }
    free(blockData);
    return 0;
}
