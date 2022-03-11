#include "AB_2.h"
// y = AA^t
void AB_2(DATA_TYPE A[N][N], DATA_TYPE A_out[N][N]) 
{
    int i, j ,k;
    DATA_TYPE buff_A[N][N]; 
    DATA_TYPE buff_A0[N][N]; 
    DATA_TYPE buff_A1[N][N];
    DATA_TYPE buff_A_out[N][N];

    lprd_1: for (i = 0; i < N; i++) {
        lprd_2: for (j = 0; j < N; j++) {
            buff_A0[i][j] = A[i][j];
            buff_A1[i][j] = A[i][j];
            buff_A_out[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) { 
            lp3: for (k = 0; k < N; k++) {
	            buff_A_out[i][j] +=  buff_A0[i][k] * buff_A1[j][k];
            }
        }
    }

    lpwr_1: for (i = 0; i < N; i++) {
        lpwr_2: for (j = 0; j < N; j++) {
            A_out[i][j] = buff_A_out[i][j];
        }
    }
}