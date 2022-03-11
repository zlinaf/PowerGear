#include "ABx_1.h"
// y = ABx
void ABx_1(DATA_TYPE A[N][N], DATA_TYPE B[N][N] ,DATA_TYPE x[N], DATA_TYPE y_out[N]) 
{
    int i, j,k;
    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_B[N][N]; 
    DATA_TYPE buff_x[N];
    DATA_TYPE buff_y_out[N];
    DATA_TYPE tmp1[N][N];

    lprd_1: for (i = 0; i < N; i++){
        buff_x[i] = x[i];
        buff_y_out[i] = 0;
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            tmp1[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
            lp3: for (k = 0; k < N; k++) {
	            tmp1[i][j] +=  buff_A[i][k] * buff_B[k][j];
            }
        }
    }

    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
            buff_y_out[i] = tmp1[i][j] * buff_x[j] + buff_y_out[i];
        } 
    }

    lpwr: for (i = 0; i < N; i++){
        y_out[i] = buff_y_out[i];
    }
}
