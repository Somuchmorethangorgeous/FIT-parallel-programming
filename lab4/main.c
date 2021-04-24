#include <mpi.h>
#include <stdlib.h>
#include <math.h>


const long double EPSILON = 10e-8;
const long double a = 10e5;

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


void sendBoundaries(const long double *phi, long double *upperBoundary, long double *lowerBoundary, MPI_Request *reqs, MPI_Request *reqr, const int rank, const int size) {
    if (rank != 0){
        MPI_Isend(phi, DimX * DimY, MPI_LONG_DOUBLE, rank - 1, 123, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(lowerBoundary, DimX * DimY, MPI_LONG_DOUBLE, rank - 1,345, MPI_COMM_WORLD, &reqr[1]);
    }
    if (rank != size - 1){
        MPI_Isend(phi + DimX * DimY * (DimZ - 1), DimX * DimY, MPI_LONG_DOUBLE, rank + 1, 345, MPI_COMM_WORLD, &reqs[1]);
        MPI_Irecv(upperBoundary, DimX * DimY, MPI_LONG_DOUBLE, rank + 1, 123, MPI_COMM_WORLD, &reqr[0]);
    }
}


void cmpOnMore(long double *op1, const long double op2) {
    *op1 = op2 > *op1 ? op2 : *op1;
}


long double calculatePhiAtPoint(const long double offsetX, const long double offsetY, const long double offsetZ){
    static const double x0 = -1.0;
    static const double y0 = -1.0;
    static const double z0 = -1.0;
    return (x0 + offsetX) * (x0 + offsetX) + (y0 + offsetY) * (y0 + offsetY) + (z0 + offsetZ) * (z0 + offsetZ);
}


long double calculatePhiOnBoundaries(long double *phi, const long double *upperBoundary, const long double *lowerBoundary,
                                     const long double hx, const long double hy, const long double hz, const int rank, const int size){
    const long double divider = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a; // coef
    long double maxDif = 0.0;
    for (int j = 1; j < DimY-1; ++j){
        for (int i = 1; i < DimX-1; ++i){
            long double difference, resultOfCalculations;
            // calculate the lower bound
            if (rank !=  0) {
                resultOfCalculations = ((phi[i+1 + DimY * j] + phi[i-1 + DimY * j]) / (hx * hx) +
                                        (phi[i + DimY * (j+1)] + phi[i + DimY * (j-1)]) / (hy * hy) +
                                        (phi[i + DimY * (j + DimX)] + lowerBoundary[i + DimY * j]) / (hz * hz) +
                                        a * phi[i + DimY * j] - 6.0) / divider;
                difference = fabsl(resultOfCalculations - phi[i + DimY * j]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * j] = resultOfCalculations;
            }
            // calculate the upper bound
            if (rank != size - 1) {
                resultOfCalculations = ((phi[i+1 + DimY * (j + DimX * (DimZ - 1))] + phi[i-1 + DimY * (j + DimX * (DimZ - 1))]) / (hx * hx) +
                                        (phi[i + DimY * (j+1 + DimX * (DimZ - 1))] + phi[i + DimY * (j-1 + DimX * (DimZ - 1))]) / (hy * hy) +
                                        (upperBoundary[i + DimY * j] + phi[i + DimY * (j + DimX * (DimZ - 2))]) / (hz * hz) +
                                        a * phi[i + DimY * (j + DimX * (DimZ - 1))] - 6.0) / divider;
                difference = fabsl(resultOfCalculations - phi[i + DimY * (j + DimX * (DimZ - 1))]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * (j + DimX * (DimZ - 1))] = resultOfCalculations;
            }
        }
    }
    return maxDif;
}


long double calculatePhiInsideArea(long double *phi, const long double hx, const long double hy, const long double hz) {
    const long double divider = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a; // coef
    long double maxDif = 0.0;
    for (int k = 1; k < DimZ - 1; ++k) {
        for (int j = 1; j < DimY - 1; ++j) {
            for (int i = 1; i < DimX - 1; ++i) {
                long double resultOfCalculations = ((phi[i+1 + DimY * (j + DimX * k)] + phi[i-1 + DimY * (j + DimX * k)]) / (hx * hx) +
                         (phi[i + DimY * (j+1 + DimX * k)] + phi[i + DimY * (j-1 + DimX * k)]) / (hy * hy) +
                         (phi[i + DimY * (j + DimX * (k + 1))] + phi[i + DimY * (j + DimX * (k - 1))]) / (hz * hz) +
                          a * phi[i + DimY * (j + DimX * k)] - 6.0) / divider;
                long double difference = fabsl(resultOfCalculations - phi[i + DimY * (j + DimX * k)]);
                cmpOnMore(&maxDif, difference);
                phi[i + DimY * (j + DimX * k)] = resultOfCalculations;
            }
        }
    }
    return maxDif;
}


void initFunctionsInside(long double *phi) {
    for (int k = 1; k < DimZ - 1; ++k){
        for (int j = 1; j < DimY - 1; ++j){
            for (int i = 1; i < DimX - 1; ++i){
                phi[i + DimY * (j + DimX * k )] = 0.0;
            }
        }
    }
}


void calculateValueOnBoundaries(long double *phi, const long double hx, const long double hy, const long double hz, const int rank, const int size){
    for (int k = 0; k < DimZ; ++k){
        for (int i = 0; i < DimX; ++i) {
            // left bound
            phi[i + DimX * DimY * k] = calculatePhiAtPoint(i * hx, 0, rank * DimZ * hz + k * hz);
            // right bound
            phi[i + DimY * ((DimY - 1) + DimX * k)] = calculatePhiAtPoint(i * hx, (DimY-1) * hy, rank * DimZ * hz + k * hz);
        }
   }
   for (int k = 0; k < DimZ; ++k) {
       for (int j = 1; j < DimY - 1; ++j) {
           // back bound
           phi[DimY * (j + DimX * k)] = calculatePhiAtPoint(0, j * hy, rank * DimZ * hz + k * hz);
           // front bound
           phi[(DimX - 1) + DimY * (j + DimX * k)] = calculatePhiAtPoint((DimX - 1) * hx, j * hy,
                                                                         rank * DimZ * hz + k * hz);
       }
   }
    //lower bound of the area
    if (rank == 0) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                phi[i + DimY * j] = calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz);
            }
        }
    }
    //upper bound of the area
    if (rank == size - 1) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                phi[i + DimY * (j + DimX * (DimZ - 1))] = calculatePhiAtPoint(i * hx, j * hy,rank * DimZ * hz + (DimZ - 1) * hz);
            }
        }
    }
}


void calculatingSizeOfCell(long double *hx, long double *hy, long double *hz, const int size) {
    const double Dx = 2.0, Dy = 2.0, Dz = 2.0;
    *hx = Dx / (DimX-1);
    *hy = Dy / (DimY-1);
    *hz = Dz / (DimZ-1);
    DimZ /= size;
}


int main(int argc, char **argv) {
    long double hx, hy, hz;
    long double maxInsideDif, maxOutsideDif, localMax, globalMax = EPSILON;
    long double localDelta = 0.0, globalDelta;
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Request reqs[2], reqr[2];
    calculatingSizeOfCell(&hx, &hy, &hz, size);
    long double *phi = (long double*) malloc(sizeof(long double) * DimX * DimY * DimZ);
    long double *upperBoundary = (long double*)malloc(sizeof(long double) * DimX * DimY);
    long double *lowerBoundary = (long double*)malloc(sizeof(long double) * DimX * DimY);

    calculateValueOnBoundaries(phi, hx, hy, hz, rank, size);
    initFunctionsInside(phi);

    //initialize upper and lower boundaries
    sendBoundaries(phi, upperBoundary, lowerBoundary, reqs, reqr, rank, size);
    waitEndOfCommunication(reqr, reqs, rank, size);

    while (globalMax >= EPSILON) {
        maxOutsideDif = calculatePhiOnBoundaries(phi, upperBoundary, lowerBoundary, hx, hy, hz, rank, size);
        sendBoundaries(phi, upperBoundary, lowerBoundary, reqs, reqr, rank, size);
        if (DimZ > 2) {
            maxInsideDif = calculatePhiInsideArea(phi, hx, hy, hz);
        } else {
            maxInsideDif = maxOutsideDif;
        }
        localMax = maxInsideDif > maxOutsideDif ? maxInsideDif : maxOutsideDif;
        MPI_Allreduce(&localMax, &globalMax, 1, MPI_LONG_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        waitEndOfCommunication(reqr, reqs, rank, size);
    }

   //calculate delta
    for(int k = 0; k < DimZ; ++k) {
        for (int j = 0; j < DimY; ++j) {
            for (int i = 0; i < DimX; ++i) {
                if (fabsl(phi[i + DimY * (j + DimX * k)] - calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz + k * hz)) > localDelta){
                    localDelta = fabsl(phi[i + DimY * (j + DimX * k)] - calculatePhiAtPoint(i * hx, j * hy, rank * DimZ * hz + k * hz));
                }
            }
        }
    }
    MPI_Allreduce(&localDelta, &globalDelta, 1, MPI_LONG_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    free(phi);
    free(upperBoundary);
    free(lowerBoundary);
    MPI_Finalize();
    return 0;
}
