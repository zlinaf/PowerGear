#include "mac_5.h"

void mac_5(DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C[M][N], DATA_TYPE A_out[M][N])
{
    int i, j;
    DATA_TYPE buff_A[M][N];
    DATA_TYPE buff_B[M][N];
    DATA_TYPE buff_C[M][N];
    
    lprd_1: for (i = 0; i < N; i++){
    	lprd_2: for(j = 0; j < N; j++){
    		buff_A[i][j] = A[i][j];
            buff_B[i][j] = B[i][j];
            buff_C[i][j] = C[i][j];
    	}
    }

    lp1: for (i = 0; i < M; i++)
        lp2: for (j = 0; j < N; j++)
	        buff_A[i][j] = buff_A[i][j] + buff_B[j][i] * buff_C[j][i]; //vs mac_1: change buff_B and buff_C reading dimension

    lpwr_1: for (i = 0; i < M; i++)
        lpwr_2: for (j = 0; j < N; j++)
            A_out[i][j] = buff_A[i][j];
}
