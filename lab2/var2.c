#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define PI 3.14159265358979323846

const int M_SIZE = 10;


double norm(const double *v){
    double tmp = 0.0;
    for (int i = 0; i < M_SIZE; ++i) {
        tmp += v[i] * v[i];
    }
    return sqrt(tmp);
}


double* solution(const double *A, const double *b, const double normB){
    static const double e = 1e-6;
    static const double t = 0.01;
    double *x = (double*)calloc(M_SIZE, sizeof(double));
    double *checkSol = (double*)malloc(sizeof(double) * M_SIZE);
    double normSol;
    do {
        normSol = 0.0;
#pragma omp parallel
        {
            #pragma omp for reduction(+:normSol)
            for (int i = 0; i < M_SIZE; ++i) {
                checkSol[i] = 0.0;
                for (int j = 0; j < M_SIZE; ++j) {
                    checkSol[i] += A[i * M_SIZE + j] * x[j];
                }
                checkSol[i] -= b[i];
                normSol += checkSol[i] * checkSol[i];
                x[i] -= t * checkSol[i];
            }
        }
    } while (sqrt(normSol) / normB >= e);
    free(checkSol);
    return x;
}


void initMatrixAndB(double *A, double *b){
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
        for (int i = 0; i < M_SIZE; ++i) {
            for (int j = 0; j < M_SIZE; ++j) {
                A[i * M_SIZE + j] = (i == j) ? 2.0 : 1.0;
            }
        }

    for (int i = 0; i < M_SIZE; ++i) {
        u[i] = sin((2 * PI * i) / M_SIZE);
        printf("%lf ", u[i]);
    } putchar('\n');

    for (int i = 0; i < M_SIZE; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < M_SIZE; ++j) {
            b[i] += A[i * M_SIZE + j] * u[j];
        }
    }
    free(u);
}


int main() {
    double *A, *b, *x;
    A = (double*)malloc(sizeof(double) * M_SIZE * M_SIZE);
    b = (double*)malloc(sizeof(double) * M_SIZE);
    initMatrixAndB(A, b);
    const double normB = norm(b);
    x = solution(A, b, normB);
#ifdef DEBUG_INFO
    printf("Answer is: ");
    for (int i = 0; i < M_SIZE; ++i){
        printf("%lf ", x[i]);
    } putchar('\n');
#endif
    free(A);
    free(b);
    free(x);
    return 0;
}

