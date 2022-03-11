#ifndef _LU_H
#define _LU_H

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

#define M 64
#define N 64

#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

void mac_2(DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C[M][N], DATA_TYPE A_out[M][N]);

#endif /* !_LU_H */
