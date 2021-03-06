#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>


const int n1 = 4;
const int n2 = 6;
const int n3 = 8;


double* initMatrix(const int X_dim, const int Y_dim){
    double *M = (double*)malloc(sizeof(double) * X_dim * Y_dim);
    for (int i = 0; i < X_dim; ++i){
        for (int j = 0; j < Y_dim; ++j){
            M[i*Y_dim + j] = rand() % 10;
        }
    }
    return M;
}


void completeMatrix(double *C, const double *blockDataC, const int sizeX, const int sizeY, const int size, MPI_Comm cart){
  MPI_Datatype resizedSubarray, blockType;
  int counts[size], displs[size], coords[size];
  const int sizes[] = {n1, n3};
  const int arrSubSizes[] = {sizeX, sizeY};
  const int arrOfStarts[] = {0, 0};

  MPI_Type_create_subarray(2, sizes, arrSubSizes, arrOfStarts, MPI_ORDER_C, MPI_DOUBLE, &blockType);
  MPI_Type_create_resized(blockType, 0, sizeof(double), &resizedSubarray);
  MPI_Type_commit(&resizedSubarray);

  for (int i = 0; i < size; ++i){
      counts[i] = 1;
      MPI_Cart_coords(cart, i, 2, coords);
      displs[i] = coords[0] * n3 * sizeX + coords[1] * sizeY;
  }

  MPI_Gatherv(blockDataC, sizeX*sizeY, MPI_DOUBLE, C, counts, displs, resizedSubarray, 0, MPI_COMM_WORLD);
  MPI_Type_free(&resizedSubarray);
}


void multiplicationInNodes(const double* blockDataA, const double *blockDataB, double *blockDataC, const int X_dimA, const int Y_dimA, const int Y_dimB){
    for (int i = 0; i < X_dimA; ++i){
        for (int j = 0; j < Y_dimB; ++j){
            blockDataC[i*Y_dimB + j] = 0.0;
            for (int k = 0; k < Y_dimA; ++k){
                blockDataC[i*Y_dimB + j] += blockDataA[i*Y_dimA + k] * blockDataB[j*n2 + k];
            }
        }
    }
}


void dataDistributionInNodes(const double *A, const double *B, double *blockDataA, double *blockDataB, const int *coords, const int sizeX, const int sizeY, MPI_Comm rowComm, MPI_Comm colComm){
    if (coords[1] == 0) {
        MPI_Scatter(A, n1 * n2 / sizeX, MPI_DOUBLE, blockDataA, n1 * n2 / sizeX, MPI_DOUBLE, 0,
                    rowComm);
    }
    if (coords[0] == 0) {
        MPI_Datatype allCol, coltype;
        MPI_Type_vector(n2, 1, n3, MPI_DOUBLE, &allCol);
        MPI_Type_commit(&allCol);
        MPI_Type_create_resized(allCol, 0, sizeof(double), &coltype);
        MPI_Type_commit(&coltype);
        MPI_Scatter(B, n3 / sizeY, coltype, blockDataB, n2 * n3 / sizeY, MPI_DOUBLE, 0,
                    colComm);
        MPI_Type_free(&allCol);
        MPI_Type_free(&coltype);
    }

    MPI_Bcast(blockDataA, n1*n2 / sizeX, MPI_DOUBLE, 0, colComm);
    MPI_Bcast(blockDataB, n2*n3 / sizeY, MPI_DOUBLE, 0, rowComm);
}


MPI_Comm initCommunicators(int *coords, const int size, MPI_Comm *colComm, MPI_Comm *rowComm){
    int dims[2] = {0, 0};
    int periods[2] = {0, 0};
    MPI_Comm comm2d;

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d);

    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    MPI_Comm_split(comm2d, coords[0], coords[1], colComm);
    MPI_Comm_split(comm2d, coords[1], coords[0], rowComm);
    return comm2d;
}


int main(int argc, char** argv) {
    double *A, *B, *C;
    double *blockDataA, *blockDataB, *blockDataC;
    MPI_Comm colComm, rowComm, comm2d;
    int coordsInGrid[2];
    int size, rank;
    int sizeX, sizeY;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        A = initMatrix(n1, n2);
        B = initMatrix(n2, n3);
        for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                printf("%lf ", A[i*n2 + j]);
            }
            putchar('\n');
        }
        putchar('\n');
        for (int i = 0; i < n2; ++i) {
            for (int j = 0; j < n3; ++j) {
                printf("%lf ", B[i*n3 + j]);
            }
            putchar('\n');
        }
        putchar('\n');
        C = (double*)malloc(sizeof(double) * n1 * n3);
    }

    comm2d = initCommunicators(coordsInGrid, size, &colComm, &rowComm);
    MPI_Comm_size(rowComm, &sizeX);
    MPI_Comm_size(colComm, &sizeY);
    blockDataA = (double*)malloc(sizeof(double) * n1 * n2 / sizeX);
    blockDataB = (double*)malloc(sizeof(double) * n2 * n3 / sizeY);
    blockDataC = (double*)malloc(sizeof(double) * n1 * n3 / (sizeX * sizeY));

    dataDistributionInNodes(A, B, blockDataA, blockDataB, coordsInGrid, sizeX, sizeY, rowComm, colComm);
    multiplicationInNodes(blockDataA, blockDataB, blockDataC, n1 / sizeX, n2, n3 / sizeY);
    completeMatrix(C, blockDataC, n1 / sizeX, n3 / sizeY, size, comm2d);

    if (rank == 0){
        for (int i = 0; i < n1; ++i){
            for (int j = 0; j < n3; ++j){
                printf("%lf ", C[i*n3 + j]);
            } putchar('\n');
        }
    }

    MPI_Comm_free(&rowComm);
    MPI_Comm_free(&colComm);
    MPI_Comm_free(&comm2d);
    MPI_Finalize();

    if (rank == 0){
        free(A);
        free(B);
        free(C);
    }
    free(blockDataA);
    free(blockDataB);
    free(blockDataC);
    return 0;
}