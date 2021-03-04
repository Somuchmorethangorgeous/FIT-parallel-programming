#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <mpi.h>


const int n1 = 4;
const int n2 = 8;
const int n3 = 4;


double* initMatrix(const int X_dim, const int Y_dim){
    double *M = (double*)malloc(sizeof(double) * X_dim * Y_dim);
    for (int i = 0; i < X_dim; ++i){
        for (int j = 0; j < Y_dim; ++j){
            M[i*Y_dim + j] = rand() % 10;
        }
    }
    return M;
}


void createCommunicators(int* coords, int *dims, const int size, MPI_Comm *colCom, MPI_Comm *rowCom){
    int periods[2] = {0, 0};
    MPI_Comm comm2d;
    int rank;

    MPI_Dims_create(size, 2, dims);
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &comm2d);

    MPI_Comm_rank(comm2d, &rank);
    MPI_Cart_get(comm2d, 2, dims, periods, coords);

    MPI_Comm_split(comm2d, coords[1], coords[0], rowCom);
    MPI_Comm_split(comm2d, coords[0], coords[1], colCom);
}

void dataDistributionOnNodes(const double *A, const double *B, double *blockDataA, double *blockDataB, MPI_Comm rowCom, MPI_Comm colCom, const int *coords, const int sizeX, const int sizeY){
    if (coords[1] == 0) {
        MPI_Scatter(A, n1 * n2 / sizeX, MPI_DOUBLE, blockDataA, n1 * n2 / sizeX, MPI_DOUBLE, 0,
                    rowCom);
    }
    if (coords[0] == 0) {
        MPI_Datatype allCol, coltype;
        MPI_Type_vector(n2, 1, n3, MPI_DOUBLE, &allCol);
        MPI_Type_commit(&allCol);
        MPI_Type_create_resized(allCol, 0, sizeof(double), &coltype);
        MPI_Type_commit(&coltype);
        MPI_Scatter(B, n3 / sizeY, coltype, blockDataB, n2 * n3 / sizeY, MPI_DOUBLE, 0,
                    colCom);
    }

    MPI_Bcast(blockDataA, n1*n2 / sizeY, MPI_DOUBLE, 0, colCom);
    MPI_Bcast(blockDataB, n2*n3 / sizeX, MPI_DOUBLE, 0, rowCom);
}


int main(int argc, char** argv) {
    double *A, *B, *C;
    double *blockDataA, *blockDataB, *blockDataC;
    MPI_Comm colCom, rowCom;
    int coords[2];
    int dims[2] = {0, 0};
    int size, rank;
    int sizeX, sizeY;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        A = initMatrix(n1, n2);
        B = initMatrix(n2, n3);
        /*for (int i = 0; i < n1; ++i) {
            for (int j = 0; j < n2; ++j) {
                printf("%lf ", A[i * n2 + j]);
            }
            putchar('\n');
        }
        putchar('\n');
        for (int i = 0; i < n2; ++i) {
            for (int j = 0; j < n3; ++j) {
                printf("%lf ", B[i * n3 + j]);
            }
            putchar('\n');
        }
        putchar('\n');*/
        C = (double *) malloc(sizeof(double) * n1 * n3);
    }

    createCommunicators(coords, dims, size, &colCom, &rowCom);
    MPI_Comm_size(colCom, &sizeX);
    MPI_Comm_size(rowCom, &sizeY);
    blockDataA = (double*) malloc(sizeof(double) * n1 * n2 / sizeX);
    blockDataB = (double*) malloc(sizeof(double) * n2 * n3 / sizeY);
    blockDataC = (double*)malloc(sizeof(double) * n1 * n3 / (sizeX * sizeY));
    dataDistributionOnNodes(A, B, blockDataA, blockDataB, rowCom, colCom, coords, sizeX, sizeY);
    /*if (coords[0] == 0 && coords[1] == 1){
       for (int i = 0; i < 16; ++i){
           printf("%lf ", blockDataA[i]);
       } putchar('\n');
       for (int i = 0; i < 16; ++i){
           printf("%lf ", blockDataB[i]);
       }
    }*/


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