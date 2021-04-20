#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

const double EPSILON = 1e-8;
const double a = 1e5;



void sendBoundaries(const double *phi, double *upperBoundary, double *lowerBoundary, const int DimX, const int DimY, const int DimZ, const int size, const int rank){
    MPI_Request reqs[2], reqr[2];
    const int numNextProc = (rank + 1) % size;
    const int numPrevProc = (rank + size - 1) % size;
    memcpy(upperBoundary, phi+DimY*DimZ*DimZ, sizeof(double)*DimX*DimY);
    memcpy(lowerBoundary, phi, sizeof(double)*DimX*DimY);
    MPI_Isend(lowerBoundary, DimX*DimY, MPI_DOUBLE, numPrevProc, 123, MPI_COMM_WORLD, &reqs[0]);
    MPI_Irecv(upperBoundary, DimX*DimY, MPI_DOUBLE, numNextProc, 123, MPI_COMM_WORLD, &reqr[0]);
   // MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
   // MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    MPI_Isend(upperBoundary, DimX*DimY, MPI_DOUBLE, numNextProc, 123, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(lowerBoundary, DimX*DimY, MPI_DOUBLE,  numPrevProc, 123, MPI_COMM_WORLD, &reqr[1]);
   // MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
   // MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
}


void iterateValueOnBoundaries(double *phi, const double *rho, const double *upperBoundary, const double *lowerBoundary, const int DimX, const int DimY, const int DimZ, const double hx, const double hy, const double hz){
    const double divider = 2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a;
    for (int i = 0; i < DimX; ++i){
        const int nextIndexI = (i + 1) % DimX;
        const int prevIndexI = (i + DimX - 1) % DimX;
        for (int j = 0; j < DimY; ++j){
            const int nextIndexJ = (j + 1) % DimY;
            const int prevIndexJ = (j + DimY - 1) % DimY;
            // calculations the lower bound
            phi[i + DimY*j] = (((phi[nextIndexI + DimY * j] - phi[prevIndexI + DimY * j]) / hx * hx +
                                (phi[i + DimY * nextIndexJ] - phi[i + DimY * prevIndexJ]) / hy * hy +
                                (phi[i + DimY * (j + DimZ)] - lowerBoundary[i + DimY * j]  / hz * hz) - rho[i + DimY * j])) / divider;
            // calculations the higher bound
            phi[i + DimY * (j + DimZ*DimZ)] =  (((phi[nextIndexI + DimY * (j + DimZ * DimZ)] - phi[prevIndexI + DimY * (j + DimZ * DimZ)]) / hx * hx +
                                                 (phi[i + DimY * (nextIndexJ + DimZ * DimZ)] - phi[i + DimY * (prevIndexJ + DimZ * DimZ)]) / hy * hy +
                                                 (upperBoundary[i + DimY * j] - phi[i + DimY * (j + DimZ * (DimZ-1))]) / hz * hz)  - rho[i + DimY * (j + DimZ * DimZ)]) / divider;
        }
    }
}


void iterateValueInsideArea(double *phi, const double *rho, const int DimX, const int DimY, const int DimZ, const double hx, const double hy, const double hz){
    const double divider = 2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a; // коэффициент перед скобкой
    for (int i = 0; i < DimX; ++i){
        const int nextIndexI = (i + 1) % DimX;
        const int prevIndexI = (i + DimX - 1) % DimX;
        for (int j = 0; j < DimY; ++j){
            const int nextIndexJ = (j + 1) % DimY;
            const int prevIndexJ = (j + DimY - 1) % DimY;
            for (int k = 1; k < DimZ - 1; ++k){
                phi[i + DimY * (j + DimZ * k)] = (((phi[nextIndexI + DimY * (j + DimZ * k)] - phi [prevIndexI + DimY * (j + DimZ * k)]) / hx*hx +
                        (phi[i + DimY * (nextIndexJ + DimZ * k)] - phi[i + DimY * (prevIndexJ + DimZ * k)]) / hy * hy +
                        (phi[i + DimY * (j + DimZ * (k+1))] - phi[i + DimY * (j + DimZ * (k-1))]) / hz * hz) - rho[i + DimY * (j + DimZ * k)]) / divider;
            }
        }
    }
}


void calculateRho(const double *phi, double *rho, const int DimX, const int DimY, const int DimZ){
    for (int i = 0; i < DimX; ++i){
        for (int j = 0; j < DimY; ++j){
            for (int k = 0; k < DimZ; ++k){
                rho[i + DimY * (j + DimZ * k)] = 6 - phi[i + DimY * (j + DimZ * k)];
            }
        }
    }
}


void initFunctions(double *phi, double *rho, const int DimX, const int DimY, const int DimZ){
    for (size_t i = 0; i < DimX * DimY * DimZ; ++i){
        phi[i] = 0.0;
        rho[i] = 6 - a * phi[i];
    }
}


void calculatingSizeOfCell(int *DimX, int *DimY, int *DimZ, double *hx, double *hy, double *hz, const int size){
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    *DimX = 20, *DimY = 20, *DimZ = 20 / size;
    *hx = Dx / *DimX;
    *hy = Dy / *DimY;
    *hz = Dz / *DimZ;
}


int main(int argc, char** argv) {
    int size, rank;
    double hx, hy, hz;
    int DimX, DimY, DimZ;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    calculatingSizeOfCell(&DimX, &DimY, &DimZ, &hx, &hy, &hz, size);
    double *rho = (double*)malloc(sizeof(double) * DimX * DimY * DimZ);
    double *phi = (double*) malloc(sizeof(double) * DimX * DimY * DimZ);
    double *upperBoundary = (double*) calloc(DimX * DimY, sizeof(double));
    double *lowerBoundary = (double*) calloc(DimX * DimY, sizeof(double));
    initFunctions(phi, rho, DimX, DimY, DimZ);
    iterateValueOnBoundaries(phi, rho, upperBoundary, lowerBoundary, DimX, DimY, DimZ, hx, hy, hz);
    //sendBoundaries(phi, upperBoundary, lowerBoundary, DimX, DimY, DimZ, size, rank);
    iterateValueInsideArea(phi, rho, DimX, DimY, DimZ, hx, hy, hz);
    calculateRho(phi, rho, DimX, DimY, DimZ);
    //wait
    int counter = 0;
    if (rank == 1) {
        for (int i = 0; i < DimX; ++i){
            for (int j = 0; j < DimY; ++j){
                for (int k = 0; k < DimZ; ++k){
                    if (fabs(phi[i + DimY * (j + DimZ * k)] + 0.000060) < 1e-5){
                        counter++;
                    }
                }
            }
        }
        printf("%d\n", counter);
    }
    free(rho);
    free(phi);
    free(upperBoundary);
    free(lowerBoundary);
    MPI_Finalize();
    return 0;
}
