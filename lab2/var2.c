#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#define PI 3.14159265358979323846

const int M_SIZE = 500;


double norm(const double *v){
    double tmp = 0.0;
    for (int i = 0; i < M_SIZE; ++i) {
        tmp += v[i] * v[i];
    }
    return sqrt(tmp);
}


bool answerIsGot(const double *A, const double *b, const double *x, const double normB){
    const double e = 1e-6;
    double *sol = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i) {
        double value = 0.0;
        for (int j = 0; j < M_SIZE; ++j) {
            value += A[i * M_SIZE + j] * x[j];
        }
        sol[i] = value - b[i];
    }
    bool result = norm(sol) / normB < e;
    free(sol);
    return result;
}


void simpleIterationMethod(const double *A, const double *b, double *x){
    const double t = 0.01;
    for (int i = 0; i < M_SIZE; ++i) {
        double value = 0.0;
        for (int j = 0; j < M_SIZE; ++j) {
            value += A[i * M_SIZE + j] * x[j];
        }
        x[i] -= t * (value - b[i]);
    }
}


double* solution(double *A, double *b, const double normB){
    double *x = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        x[i] = 0.0;
    }
    bool isFinish = false;
#pragma omp parallel
    {
        do {
            simpleIterationMethod(A, b, x);
            isFinish = answerIsGot(A, b, x, normB);
        } while (!isFinish);
    }
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
    }

    for (int i = 0; i < M_SIZE; ++i) {
        b[i] = 0.0;
        for (int j = 0; j < M_SIZE; ++j) {
            b[i] += A[i * M_SIZE + j] * u[j];
        }
    }
#ifdef DEBUG_INFO
    for (int i = 0; i < M_SIZE; ++i){
        printf("%lf ", u[i]);
    } putchar('\n');
#endif
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

