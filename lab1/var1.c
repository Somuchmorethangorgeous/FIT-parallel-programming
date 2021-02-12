#include <stdio.h>
#include <stdbool.h>
#include <malloc.h>
#include <math.h>

static int M_SIZE;

void printVector(double *v){
    for (int i = 0; i < M_SIZE; ++i){
        printf("%lf ", v[i]);
    }
    putchar('\n');
}

void printMatrix(double *A){
    for (int i = 0; i < M_SIZE; ++i) {
        for (int j = 0; j < M_SIZE; ++j) {
            printf("%lf ", A[i*M_SIZE + j]);
        }
        putchar('\n');
    }
}

double norm(const double *v){
    double tmp = 0;
    for (int i = 0; i < M_SIZE; ++i){
        tmp += v[i] * v[i];
    }
    return sqrt(tmp);
}

bool answerIsGot(const double *A, const double *b, const double *x){
    const double e = pow(10, -6);
    double *sol = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i) {
        double value = 0;
        for (int j = 0; j < M_SIZE; ++j) {
            value += A[i * M_SIZE + j] * x[j];
        }
        sol[i] = value - b[i];
    }
    if (norm(sol) / norm (b) < e){
        free(sol);
        return true;
    } else {
        free(sol);
        return false;
    }
}


void simpleIterationMethod(const double *A, const double *b, double *x){
    const double t = 0.01;
    for (int i = 0; i < M_SIZE; ++i){
        double value = 0;
        for (int j = 0; j < M_SIZE; ++j){
            value += A[i*M_SIZE + j] * x[j];
        }
        x[i] -= t * (value - b[i]);
    }
}

double* solution(double *A, double *b){
    double *x = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        x[i] = 0;
    }
    do {
        simpleIterationMethod(A,b,x);
    } while (!answerIsGot(A,b,x));
    return x;
}


double* initMatrixAndB(double *A){
    double *u = (double*)malloc(sizeof(double) * M_SIZE);
    for (int i = 0; i < M_SIZE; ++i){
        for (int j = 0; j < M_SIZE; ++j){
            if (i == j){
                A[i*M_SIZE+j] = 2.0;
            } else {
                A[i*M_SIZE+j] = 1.0;
            }
        }
    }
    for (int j = 0; j < M_SIZE; ++j){
        u[j] = sin((2*M_PI*j) / M_SIZE);
    }
    double *b = (double*)malloc(sizeof(double) * M_SIZE);
    for (int m = 0; m < M_SIZE; ++m){
        b[m] = 0;
        for (int n = 0; n < M_SIZE; ++n){
            b[m] += A[m*M_SIZE + n] * u[n];
        }
    }
    printVector(u);
    putchar('\n');
    free(u);
    return b;
}

int main() {
    scanf("%d", &M_SIZE);
    double *A = (double*)malloc(sizeof(double) * M_SIZE * M_SIZE);
    double *b, *x;
    b = initMatrixAndB(A);
    x = solution(A, b);
    printVector(x);
    free(A);
    free(b);
    free(x);
    return 0;
}
