#include "AB_1.h"
// c=AB
void AB_1(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C_out[N][N])
{
    int i, j, k;
    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_B[N][N];
    DATA_TYPE buff_C_out[N][N];

    lprd_1: for (i = 0; i < N; i++) {
        lprd_2: for (j = 0; j < N; j++) {
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C_out[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) { 
            lp3: for (k = 0; k < N; k++) {
	            buff_C_out[i][j] +=  buff_A[i][k] * buff_B[k][j];
            }
        }
    }

    lpwr_1: for (i = 0; i < N; i++){
        lpwr_2: for (j = 0; j < N; j++){
            C_out[i][j] = buff_C_out[i][j];
        }
    }
}