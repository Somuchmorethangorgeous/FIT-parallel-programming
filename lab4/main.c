#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>


const double EPSILON = 10e-8;
const double a = 10e5;

const int DimX = 16;
const int DimY = 16;
int DimZ = 16;


void waitEndOfCommunication(MPI_Request *reqr, MPI_Request *reqs, const int rank, const int size){
    if (rank != 0){
        MPI_Wait(&reqs[0], MPI_STATUS_IGNORE);
        MPI_Wait(&reqr[1], MPI_STATUS_IGNORE);
    }
    if (rank != size - 1){
        MPI_Wait(&reqs[1], MPI_STATUS_IGNORE);
        MPI_Wait(&reqr[0], MPI_STATUS_IGNORE);
    }
}


void sendBoundaries(const double *phi, double *upperBoundary, double *lowerBoundary, MPI_Request *reqs, MPI_Request *reqr, const int rank, const int size) {
    if (rank != 0){
        MPI_Isend(&phi[0], DimX * DimY, MPI_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(lowerBoundary, DimX * DimY, MPI_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, &reqr[1]);
    }
    if (rank != size - 1){
        MPI_Isend(&phi[DimX * DimY * (DimZ - 1)], DimX * DimY, MPI_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(upperBoundary, DimX * DimY, MPI_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, &reqr[0]);
    }
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


double calculatePhiOnBoundaries(double *phi, const double *rho, const double *upperBoundary, const double *lowerBoundary,
                                 const double hx, const double hy, const double hz, const int rank, const int size){
    const double divider = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a; // coef
    double maxDif = 0.0;
    for (int j = 0; j < DimY; ++j){
        const int nextIndexJ = (j + 1) % DimY;
        const int prevIndexJ = (j + DimY - 1) % DimY;
        for (int i = 0; i < DimX; ++i){
            double difference, resultOfCalculations;
            const int nextIndexI = (i + 1) % DimX;
            const int prevIndexI = (i + DimX - 1) % DimX;
            // calculate the lower bound
            if (rank !=  0) {
                resultOfCalculations = ((phi[nextIndexI + DimY * j] - phi[prevIndexI + DimY * j]) / (hx * hx) +
                                        (phi[i + DimY * nextIndexJ] - phi[i + DimY * prevIndexJ]) / (hy * hy) +
                                        (phi[i + DimY * (j + DimX)] - lowerBoundary[i + DimY * j]) / (hz * hz) -
                                        rho[i + DimY * j]) / divider;
                difference = fabs(resultOfCalculations - phi[i + DimY * j]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * j] = resultOfCalculations;
            }
            // calculate the higher bound
            if (rank != size - 1) {
                resultOfCalculations = ((phi[nextIndexI + DimY * (j + DimX * (DimZ - 1))] - phi[prevIndexI + DimY * (j + DimX * (DimZ - 1))]) / (hx * hx) +
                                        (phi[i + DimY * (nextIndexJ + DimX * (DimZ - 1))] - phi[i + DimY * (prevIndexJ + DimX * (DimZ - 1))]) / (hy * hy) +
                                        (upperBoundary[i + DimY * j] - phi[i + DimY * (j + DimX * (DimZ - 2))]) / (hz * hz) -
                                        rho[i + DimY * (j + DimX * (DimZ - 1))]) / divider;
                difference = fabs(resultOfCalculations - phi[i + DimY * (j + DimX * (DimZ - 1))]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * (j + DimX * (DimZ - 1))] = resultOfCalculations;
            }
        }
    }
    return maxDif;
}


double calculatePhiInsideArea(double *phi, const double *rho, const double hx, const double hy, const double hz) {
    const double divider = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a; // coef
    double maxDif = 0.0;
    for (int k = 1; k < DimZ-1; ++k) {
        for (int j = 1; j < DimY - 1; ++j) {
            for (int i = 1; i < DimX - 1; ++i) {
                double resultOfCalculations = ((phi[i+1 + DimY * (j + DimX * k)] - phi[i-1 + DimY * (j + DimX * k)]) / (hx * hx) +
                         (phi[i + DimY * (j+1 + DimX * k)] - phi[i + DimY * (j-1 + DimX * k)]) / (hy * hy) +
                         (phi[i + DimY * (j + DimX * (k + 1))] - phi[i + DimY * (j + DimX * (k - 1))]) / (hz * hz) -
                         rho[i + DimY * (j + DimX * k)]) / divider;
                double difference = fabs(resultOfCalculations - phi[i + DimY * (j + DimX * k)]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * (j + DimX * k)] = resultOfCalculations;
            }
        }
    }
    return maxDif;
}


void calculateRho(const double *phi, double *rho) {
    for (int k = 0; k < DimZ; ++k) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                rho[i + DimY * (j + DimX * k)] = 6.0 - a * phi[i + DimY * (j + DimX * k)];
            }
        }
    }
}


void initFunctionsInside(double *phi, double *rho) {
    for (int k = 1; k < DimZ - 1; ++k){
        for (int j = 1; j < DimY - 1; ++j){
            for (int i = 1; i < DimX - 1; ++i){
                phi[i + DimY * (j + DimX * k )] = 0.0;
            }
        }
    }
    calculateRho(phi, rho);
}


void calculateValueOnBoundaries(double *phi, const double hx, const double hy, const double hz, const int rank){
    for (int j = 0; j < DimY; ++j){
        for (int i = 0; i < DimX; ++i){
            //lower bound
            phi[i + DimY * j] = calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz);
            //higher bound
            phi[i + DimY * (j + DimX * (DimZ-1))] = calculatePhiAtPoint(i * hx, j * hy,  rank * DimZ * hz + (DimZ-1) * hz);
        }
    }
    for (int k = 1; k < DimZ - 1; ++k){
        for (int i = 0; i < DimX; ++i) {
            // left bound
            phi[i + DimX * DimY * k] = calculatePhiAtPoint(i * hx, 0, rank * DimZ * hz + k * hz);
            // right bound
            phi[i + DimY * ((DimY - 1) + DimX * k)] = calculatePhiAtPoint(i * hx, (DimY-1) * hy, rank * DimZ * hz + k * hz);
        }
   }
   for (int k = 1; k < DimZ - 1; ++k){
       for (int j = 0; j < DimY; ++j){
           // back bound
           phi[DimY * (j + DimX * k)] = calculatePhiAtPoint(0, j * hy, rank * DimZ * hz + k * hz);
           // front bound
           phi[(DimX - 1) + DimY * (j + DimX * k)] = calculatePhiAtPoint((DimX-1)*hx, j * hy, rank * DimZ * hz + k * hz);
       }
   }
}


void calculatingSizeOfCell(double *hx, double *hy, double *hz, const int size) {
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    *hx = Dx / (DimX-1);
    *hy = Dy / (DimY-1);
    *hz = Dz / (DimZ-1);
    DimZ /= size;
}


int main(int argc, char **argv) {
    double hx, hy, hz;
    double maxInsideDif, maxOutsideDif, localMax, globalMax;
    double localDelta = 0.0, globalDelta;
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request reqs[2], reqr[2];
    calculatingSizeOfCell(&hx, &hy, &hz, size);
    double *rho = (double*) malloc(sizeof(double) * DimX * DimY * DimZ);
    double *phi = (double*) malloc(sizeof(double) * DimX * DimY * DimZ);
    double *upperBoundary = (double*)malloc(sizeof(double) * DimX * DimY);
    double *lowerBoundary = (double*)malloc(sizeof(double) * DimX * DimY);

    calculateValueOnBoundaries(phi, hx, hy, hz, rank);
    initFunctionsInside(phi, rho);

    //initialize upper and lower boundary
    sendBoundaries(phi, upperBoundary, lowerBoundary, reqs, reqr, rank, size);
    waitEndOfCommunication(reqr, reqs, rank, size);


    while (true) {
        maxOutsideDif = calculatePhiOnBoundaries(phi, rho, upperBoundary, lowerBoundary, hx, hy, hz, rank, size);
        sendBoundaries(phi, upperBoundary, lowerBoundary, reqs, reqr, rank, size);
        if (DimZ > 1) {
            maxInsideDif = calculatePhiInsideArea(phi, rho, hx, hy, hz);
        } else {
            maxInsideDif = 0.0;
        }
        calculateRho(phi, rho);
        localMax = maxInsideDif > maxOutsideDif ? maxInsideDif : maxOutsideDif;
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        waitEndOfCommunication(reqr, reqs, rank, size);
        if (globalMax < EPSILON)
            break;
    }

   //counting delta
    for(int k = 0; k < DimZ; ++k) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                if (fabs(phi[i + DimY * (j + DimX * k)] - calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz + k * hz)) > localDelta){
                    localDelta = fabs(phi[i + DimY * (j + DimX * k)] - calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz + k * hz));
                }
            }
        }
    }
    MPI_Allreduce(&localDelta, &globalDelta, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    free(rho);
    free(phi);
    free(upperBoundary);
    free(lowerBoundary);
    MPI_Finalize();
    return 0;
}
