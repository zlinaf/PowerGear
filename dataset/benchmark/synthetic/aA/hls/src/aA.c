#include "aA.h"
// A_out=aA
void aA(DATA_TYPE alpha, DATA_TYPE A[N][N], DATA_TYPE A_out[N][N]) 
{
    int i, j, k;

    DATA_TYPE buff_A[N][N]; 
    DATA_TYPE buff_A_out[N][N];

    lprd_1: for (i = 0; i < N; i++){
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
            buff_A_out[i][j] = alpha * buff_A[i][j];
        }
    }    

    lpwr_1: for (i = 0; i < N; i++){
        lpwr_2: for (j = 0; j < N; j++){
            A_out[i][j] =  buff_A_out[i][j];
        }
    }
}