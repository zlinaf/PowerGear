#include "ABC.h"
// D = ABC
void ABC(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N],DATA_TYPE D_out[N][N]) 
{
    int i, j, k;

    DATA_TYPE buff_A[N][N]; 
    DATA_TYPE buff_B[N][N]; 
    DATA_TYPE buff_C[N][N];
    DATA_TYPE buff_D_out[N][N];
    DATA_TYPE tmp[N][N];

    lprd_1: for (i = 0; i < N; i++){
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C[i][j] = C[i][j];
            tmp[i][j] = 0;
            buff_D_out[i][j] = 0;
        }
    }
    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
            lp3: for (k = 0; k < N; k++) {
	            tmp[i][j] +=  buff_A[i][k] * buff_B[k][j];
            }
        }
    }

    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
            lp6: for (k = 0; k < N; k++) {
	            buff_D_out[i][j] += tmp[i][k] * buff_C[k][j];
            }
        }
    }

    lpwr_1: for (i = 0; i < N; i++) {
        lpwr_2: for (j = 0; j < N; j++) {
            D_out[i][j] = buff_D_out[i][j];
        }
    }
}