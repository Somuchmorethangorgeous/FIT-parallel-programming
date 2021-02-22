#include <stdio.h>
#include <stdbool.h>
#include <mpi.h>
#include <malloc.h>
#include <math.h>


const int M_SIZE = 10;


double norm(const double *v,  const int len){
    double localRes, globalRes;
    localRes = 0.0;
    for (int i = 0; i < len; ++i){
        localRes += v[i] * v[i];
    }
    MPI_Allreduce(&localRes, &globalRes, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(globalRes);
}


bool answerIsGot(const double *blockData, const double *partVecB, const double *partVecX, const int *colsForEachProc, int rank, const double normB){
    const double e = pow(10, -6);
    double *mulVec = (double*)malloc(sizeof(double) * M_SIZE);
    double *sol = (double*)malloc(sizeof(double) * colsForEachProc[rank]);
    for (int i = 0; i < M_SIZE; ++i){
        mulVec[i] = 0;
    }

    for (int i = 0; i < colsForEachProc[rank]; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            mulVec[j] += blockData[i*M_SIZE + j] * partVecX[i];
        }
    }

    MPI_Reduce_scatter(mulVec, sol, colsForEachProc, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < colsForEachProc[rank]; ++i){
        sol[i] -= partVecB[i];
    }

    bool result = norm(sol, colsForEachProc[rank]) / normB < e;

    free(sol);
    free(mulVec);
    return result;
}


void simpleIterationMethod(const double *blockData, const double *partVecB, double *partVecX, const int *numCols, int rank){
    double *mulVec = (double*)malloc(sizeof(double) * M_SIZE);
    double *resVec = (double*)malloc(sizeof(double) * numCols[rank]);
    const double t = 0.01;

    for (int i = 0; i < M_SIZE; ++i){
        mulVec[i] = 0.0;
    }
    for (int i = 0; i < numCols[rank]; ++i) {
        for (int j = 0; j < M_SIZE; ++j) {
            mulVec[j] += blockData[i*M_SIZE+j] * partVecX[i];
        }
    }

    MPI_Reduce_scatter(mulVec, resVec, numCols, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < numCols[rank]; ++i){
        partVecX[i] -= t * (resVec[i] - partVecB[i]);
    }
    free(mulVec);
    free(resVec);
}


double* solution(const double *blockData, const double *partVecB, int *dataVec, int rank, const double normB){
   double *partVecX = (double*)malloc(sizeof(double) * dataVec[rank]);
    for (int i = 0; i < dataVec[rank]; ++i){
        partVecX[i] = 0.0;
    }
    bool isFinish = false;
    do {
        simpleIterationMethod(blockData, partVecB, partVecX, dataVec, rank);
        isFinish = answerIsGot(blockData, partVecB, partVecX, dataVec, rank, normB);
    } while (!isFinish);
    return partVecX;
}


void initMatrixAndB(double *blockData, double *partVecB, const int *colsForEachProc, const int *shiftForEachProc, int rank){
    const int shift = shiftForEachProc[rank];
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < colsForEachProc[rank]; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            blockData[i*M_SIZE+j] = (i + shift == j ) ?  2.0 : 1.0;
        }
        u[i+shift] = sin((2*M_PI*(i+shift) / M_SIZE));
    }

    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, u, colsForEachProc, shiftForEachProc, MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < colsForEachProc[rank]; ++i){
        partVecB[i] = 0.0;
        for (int j = 0; j < M_SIZE; ++j){
            partVecB[i] += blockData[i*M_SIZE + j] * u[j];
        }
    }
    free(u);
}


double* cutMatrix(double *partVecB, int *dataVec, int *shiftVec, int rank){
    double *blockData = (double*)malloc(sizeof(double) * dataVec[rank] * M_SIZE);
    initMatrixAndB(blockData, partVecB, dataVec, shiftVec, rank);
    return blockData;
}


void dataDistribution(int *dataVec, int *shiftVec, int numProcs){
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
    double *partVecB, *partVecX, *blockData;
    int *dataVec, *shiftVec;
    int size, rank;
    double normB;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    dataVec = (int*)malloc(sizeof(int) * size);
    shiftVec = (int*)malloc(sizeof(int) * size);
    dataDistribution(dataVec, shiftVec, size);

    partVecB = (double*)malloc(sizeof(double) * dataVec[rank]);
    blockData = cutMatrix(partVecB, dataVec, shiftVec, rank);
    normB = norm(partVecB, dataVec[rank]);

    partVecX = solution(blockData, partVecB, dataVec, rank, normB);

    MPI_Finalize();
    free(partVecB);
    free(partVecX);
    free(blockData);
    free(dataVec);
    free(shiftVec);
    return 0;
}
