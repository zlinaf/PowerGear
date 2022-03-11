#ifndef _G_H
#define _G_H

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define N 64

#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

void mvm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE u1[N], DATA_TYPE v1[N], DATA_TYPE u2[N], DATA_TYPE v2[N],
    DATA_TYPE y[N], DATA_TYPE z[N], DATA_TYPE A_out[N][N], DATA_TYPE x_out[N], DATA_TYPE w_out[N]);

#endif /* !_G_H */
