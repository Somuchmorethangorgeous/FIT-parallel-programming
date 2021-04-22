#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


const double EPSILON = 2e-6;
const double a = 1e6;


void waitEndOfCommunication(MPI_Request *reqr, MPI_Request *reqs) {
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
    MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
    MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
}


void sendBoundaries(const double *phi, double *upperBoundary, double *lowerBoundary, const int DimX, const int DimY,
                    const int DimZ, const int size, const int rank, MPI_Request *reqs, MPI_Request *reqr) {
    const int numNextProc = (rank + 1) % size;
    const int numPrevProc = (rank + size - 1) % size;
    memcpy(upperBoundary, phi + DimX * DimY * (DimZ-1), sizeof(double) * DimX * DimY);
    memcpy(lowerBoundary, phi, sizeof(double) * DimX * DimY);
    MPI_Isend(lowerBoundary, DimX * DimY, MPI_DOUBLE, numPrevProc, 123, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(upperBoundary, DimX * DimY, MPI_DOUBLE, numNextProc, 123, MPI_COMM_WORLD, &reqs[1]);
    MPI_Irecv(upperBoundary, DimX * DimY, MPI_DOUBLE, numNextProc, 123, MPI_COMM_WORLD, &reqr[0]);
    MPI_Irecv(lowerBoundary, DimX * DimY, MPI_DOUBLE, numPrevProc, 123, MPI_COMM_WORLD, &reqr[1]);
}


void cmpOnMore(double *op1, const double op2) {
    *op1 = op2 > *op1 ? op2 : *op1;
}


double calculatePhiAtPoint(const double offsetX, const double offsetY, const double offsetZ){
    static const double x0 = -1.0;
    static const double y0 = -1.0;
    static const double z0 = -1.0;
    return (x0 + offsetX) * (x0 + offsetX) + (y0 + offsetY) * (y0 + offsetY) + (z0 + offsetZ) * (z0 + offsetZ);
}


double
iterateValueOnBoundaries(double *phi, const double *rho, const double *upperBoundary, const double *lowerBoundary,
                         const int DimX, const int DimY, const int DimZ, const double hx, const double hy,
                         const double hz) {
    const double divider = 2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a;
    double resultOfCalculations;
    double maxDif = 0.0;
    for (int j = 0; j < DimY; ++j){
        const int nextIndexJ = (j + 1) % DimY;
        const int prevIndexJ = (j + DimY - 1) % DimY;
        for (int i = 0; i < DimX; ++i){
            const int nextIndexI = (i + 1) % DimX;
            const int prevIndexI = (i + DimX - 1) % DimX;
            // calculations the lower bound
            resultOfCalculations = ((phi[nextIndexI + DimY * j] - phi[prevIndexI + DimY * j]) / (hx * hx) +
                                    (phi[i + DimY * nextIndexJ] - phi[i + DimY * prevIndexJ]) / (hy * hy) +
                                    (phi[i + DimY * j] - lowerBoundary[i + DimY * j]) / (hz * hz) -
                                    rho[i + DimY * j]) / divider;
            cmpOnMore(&maxDif, fabs(resultOfCalculations - phi[i + DimY * j]));
            phi[i + DimY * j] = resultOfCalculations;
            // calculations the higher bound
            resultOfCalculations = ((phi[nextIndexI + DimY * (j + DimX * DimZ)] - phi[prevIndexI + DimY * (j + DimX * DimZ)]) / (hx * hx) +
                     (phi[i + DimY * (nextIndexJ + DimX * DimZ)] - phi[i + DimY * (prevIndexJ + DimX * DimZ)]) / (hy * hy) +
                     (upperBoundary[i + DimY * j] - phi[i + DimY * (j + DimX * DimZ)]) / (hz * hz) - rho[i + DimY * (j + DimX * DimZ)]) / divider;
            cmpOnMore(&maxDif, fabs(resultOfCalculations - phi[i + DimY * (j + DimX * DimZ)]));
            phi[i + DimY * (j + DimX * DimZ)] = resultOfCalculations;
        }
    }
    return maxDif;
}


double
iterateValueInsideArea(double *phi, const double *rho, const int DimX, const int DimY, const int DimZ, const double hx,
                       const double hy, const double hz) {
    const double divider = 2 / (hx * hx) + 2 / (hy * hy) + 2 / (hz * hz) + a; // коэффициент перед скобкой
    double resultOfCalculations;
    double maxDif = 0.0;
    for (int k = 1; k < DimZ; ++k) {
        for (int j = 0; j < DimY; ++j) {
            const int nextIndexJ = (j + 1) % DimY;
            const int prevIndexJ = (j + DimY - 1) % DimY;
            for (int i = 0; i < DimX; ++i) {
                const int nextIndexI = (i + 1) % DimX;
                const int prevIndexI = (i + DimX - 1) % DimX;
                resultOfCalculations = ((phi[nextIndexI + DimY * (j + DimX * k)] - phi[prevIndexI + DimY * (j + DimX * k)]) / (hx * hx) +
                         (phi[i + DimY * (nextIndexJ + DimX * k)] - phi[i + DimY * (prevIndexJ + DimX * k)]) / (hy * hy) +
                         (phi[i + DimY * (j + DimX * (k + 1))] - phi[i + DimY * (j + DimX * (k - 1))]) / (hz * hz) -
                         rho[i + DimY * (j + DimX * k)]) / divider;
                cmpOnMore(&maxDif, fabs(resultOfCalculations - phi[i + DimY * (j + DimX * k)]));
                phi[i + DimY * (j + DimX * k)] = resultOfCalculations;
            }
        }
    }
    return maxDif;
}


void calculateRho(const double *phi, double *rho, const int DimX, const int DimY, const int DimZ) {
    for (int k = 0; k < DimZ; ++k) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                rho[i + DimY * (j + DimX * k)] = 6 - a * phi[i + DimY * (j + DimX * k)];
            }
        }
    }
}


void calculateValueOnBoundaries(double *phi, const int DimX, const int DimY, const int DimZ, const double hx, const double hy, const double hz, const int rank){
    for (int j = 0; j < DimY; ++j){
        for (int i = 0; i < DimX; ++i){
            //lower bound
            phi[i + DimY * j] = calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz);
            //higher bound
            phi[i + DimY * (j + DimX * (DimZ-1))] = calculatePhiAtPoint(i * hx, j * hy,  rank * DimZ * hz + (DimZ - 1) * hz);
        }
    }
    for (int k = 1; k < DimZ; ++k){
        for (int i = 0; i < DimX; ++i) {
            // left bound
            phi[i + DimX * DimY * k] = calculatePhiAtPoint(i * hx, 0, rank * DimZ * hz + k * hz);
            // right bound
            phi[i + DimY * ((DimY - 1) + DimX * k)] = calculatePhiAtPoint(i * hx, (DimY - 1) * hy, rank * DimZ * hz + k * hz);
        }
   }
   for (int k = 1; k < DimZ; ++k){
       for (int j = 0; j < DimY; ++j){
           // back bound
           phi[DimY * (j + DimX * k)] = calculatePhiAtPoint(0, j * hy, rank * DimZ * hz + k * hz);
           // front bound
           phi[(DimX - 1) + DimY * (j + DimX * k)] = calculatePhiAtPoint(hx * (DimX -1), j * hy, rank * DimZ * hz + k * hz);
       }
   }
}


void initFunctions(double *phi, double *rho, const int DimX, const int DimY, const int DimZ) {
    for (size_t i = 0; i < DimX * DimY * DimZ; ++i) {
        phi[i] = 0.0;
        rho[i] = 6 - a * phi[i];
    }
}


void calculatingSizeOfCell(int *DimX, int *DimY, int *DimZ, double *hx, double *hy, double *hz, const int size) {
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    *DimX = 16, *DimY = 16, *DimZ = 16 / size;
    *hx = Dx / (*DimX - 1);
    *hy = Dy / (*DimY - 1);
    *hz = Dz / (*DimZ - 1);
}


int main(int argc, char **argv) {
    int size, rank;
    double hx, hy, hz;
    double maxInside, maxOutside, localMax, globalMax;
    int DimX, DimY, DimZ;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request reqs[2], reqr[2];
    calculatingSizeOfCell(&DimX, &DimY, &DimZ, &hx, &hy, &hz, size);
    double *rho = (double *) malloc(sizeof(double) * DimX * DimY * DimZ);
    double *phi = (double *) malloc(sizeof(double) * DimX * DimY * DimZ);
    double *upperBoundary = (double*)calloc(DimX * DimY, sizeof(double));
    double *lowerBoundary = (double*)calloc(DimX * DimY, sizeof(double));
    initFunctions(phi, rho, DimX, DimY, DimZ);
    calculateValueOnBoundaries(phi, DimX, DimY, DimZ, hx, hy, hz, rank);
    while (true) {
        maxOutside = iterateValueOnBoundaries(phi, rho, upperBoundary, lowerBoundary, DimX, DimY, DimZ, hx, hy, hz);
        sendBoundaries(phi, upperBoundary, lowerBoundary, DimX, DimY, DimZ, size, rank, reqs, reqr);
        maxInside = iterateValueInsideArea(phi, rho, DimX, DimY, DimZ, hx, hy, hz);
        calculateRho(phi, rho, DimX, DimY, DimZ);
        localMax = maxInside < maxOutside ? maxInside : maxOutside;
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
       // if (rank == 0)
       //     printf("%lf %lf %lf\n", maxInside, maxOutside, globalMax);
        waitEndOfCommunication(reqr, reqs);
        if (globalMax < EPSILON)
            break;
    }
    if (rank == 0) {
        /*(for (int k = 0; k < DimZ; ++k) {
            for (int j = 0; j < DimY; ++j) {
                for (int i = 0; i < DimX; ++i) {
                    printf("%lf ", phi[i + DimY * (j + DimZ * k)]);
                }
            }
        }*/
        printf("%lf\n", phi[0]);
    }
    free(rho);
    free(phi);
    free(upperBoundary);
    free(lowerBoundary);
    MPI_Finalize();
    return 0;
}
