#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846

const int M_SIZE = 10;


double norm(const double *v) {
    double tmp = 0.0;
    for (int i = 0; i < M_SIZE; ++i) {
        tmp += v[i] * v[i];
    }
    return sqrt(tmp);
}


double *solution(const double *A, const double *b, const double normB) {
    static const double e = 1e-5;
    static const double t = 0.01;
    double *x = (double*)calloc(M_SIZE, sizeof(double));
    double checkSol[M_SIZE];
    double normSol = 0., value;
    bool isFinish = false;
#pragma omp parallel private(value)
    {
        while (!isFinish) {
#pragma omp for
            for (int i = 0; i < M_SIZE; ++i) {
                value = 0.0;
                for (int j = 0; j < M_SIZE; ++j) {
                    value += A[i * M_SIZE + j] * x[j];
                }
                x[i] -= t * (value - b[i]);
            }
#pragma omp for reduction(+:normSol)
            for (int i = 0; i < M_SIZE; ++i) {
                value = 0.0;
                for (int j = 0; j < M_SIZE; ++j) {
                    value += A[i * M_SIZE + j] * x[j];
                }
                normSol += (value - b[i]) * (value - b[i]);
            }
#pragma omp single
            {
                isFinish = sqrt(normSol) / normB < e;
                normSol = 0.0;
            }
        }
    }
    return x;
}


void initMatrixAndB(double *A, double *b) {
    double u[M_SIZE];
    for (int i = 0; i < M_SIZE; ++i) {
        for (int j = 0; j < M_SIZE; ++j) {
            A[i * M_SIZE + j] = (i == j) ? 2.0 : 1.0;
        }
    }

    for (int i = 0; i < M_SIZE; ++i) {
        u[i] = sin((2 * PI * i) / M_SIZE);
        printf("%lf ", u[i]);
    }
    putchar('\n');

    for (int i = 0; i < M_SIZE; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < M_SIZE; ++j) {
            b[i] += A[i * M_SIZE + j] * u[j];
        }
    }
}


int main() {
    double *A = (double *) malloc(sizeof(double) * M_SIZE * M_SIZE);
    double b[M_SIZE];
    initMatrixAndB(A, b);
    const double normB = norm(b);
    double *x = solution(A, b, normB);
#ifdef DEBUG_INFO
    printf("Answer is: ");
    for (int i = 0; i < M_SIZE; ++i) {
        printf("%lf ", x[i]);
    }
    putchar('\n');
#endif
    free(x);
    free(A);
    return 0;
}

