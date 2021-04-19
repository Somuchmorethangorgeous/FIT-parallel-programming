#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>

const double EPSILON = 1e-8;
const double a = 1e5;



void iterateValueOnBounaries(double *phi, double *rho, const int DimX, const int DimY, const int DimZ, const double hx, const double hy, const double hz, const int rank, const int size){
    MPI_Request reqs[2], reqr[2];
    double upperBoundary[DimX * DimY];
    double lowerBoundary[DimX * DimY];
    const double k = 1.0 /(2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a);
    memcpy(upperBoundary, phi+DimY*DimZ*(DimZ-1), sizeof(double)*DimX*DimY);
    memcpy(lowerBoundary, phi, sizeof(double)*DimX*DimY);
    MPI_Irecv(upperBoundary, DimX*DimY, MPI_DOUBLE, (rank + 1) % size, 123, MPI_COMM_WORLD, &reqr[0]);
    MPI_Isend(lowerBoundary, DimX*DimY, MPI_DOUBLE, (rank  + size - 1) % size, 123, MPI_COMM_WORLD, &reqs[0]);
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    MPI_Irecv(lowerBoundary, DimX*DimY, MPI_DOUBLE, (rank + size - 1) % size, 123, MPI_COMM_WORLD, &reqr[1]);
    MPI_Isend(upperBoundary, DimX*DimY, MPI_DOUBLE, (rank + 1) % size, 123, MPI_COMM_WORLD, &reqs[1]);
    MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
    if (rank == 0) {
        for (int i = 0; i < DimX * DimY; ++i) {
            printf("%lf ", upperBoundary[i]);
        }
    }

}

void iterateValueInsideArea(double *rho, double *phi, const double hx, const double hy, const double hz){
    const double k = 1.0 /(2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a); // коэффициент перед скобкой


}

void initFunctions(double *phi, double *rho, const int DimX, const int DimY, const int DimZ, const int rank){
    for (size_t i = 0; i < DimX*DimY*DimZ; ++i){
        phi[i] = rank;
        rho[i] = 6 - a * phi[i];
    }
}


void calculatingSizeOfCell(int *DimX, int *DimY, int *DimZ, double *hx, double *hy, double *hz, const int size){
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    *DimX = 20 / size, *DimY = 20 / size, *DimZ = 20 / size;
    *hx = Dx / *DimX;
    *hy = Dy / *DimY;
    *hz = Dz / *DimZ;
}


int main(int argc, char** argv) {
    int size, rank;
    double hx, hy,hz;
    int DimX,DimY,DimZ;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    calculatingSizeOfCell(&DimX, &DimY, &DimZ, &hx, &hy, &hz, size);
    double *rho = (double*)malloc(sizeof(double) * DimX * DimY * DimZ);
    double *phi = (double*) malloc(sizeof(double) * DimX * DimY * DimZ);
    initFunctions(phi, rho, DimX, DimY, DimZ, rank);
    iterateValueOnBounaries(phi, rho, DimX, DimY, DimZ, hx, hy, hz, rank, size);
    free(rho);
    free(phi);
    MPI_Finalize();
    return 0;
}
