#include "xy.h"

void xy(DATA_TYPE x[N], DATA_TYPE y[N], DATA_TYPE A_out[N][N])
{
    int i, j;
    DATA_TYPE buff_x[N];
    DATA_TYPE buff_y[N];
    DATA_TYPE buff_A_out[N][N];

    lprd_1: for (i = 0; i < N; i++){
        buff_x[i] = x[i];
        buff_y[i] = y[i];
        lprd_2: for (j = 0; j < N; j++){
            buff_A_out[i][j] = 0;
        }
    }

    lp1: for (i = 0; i < N; i++)
        lp2: for (j = 0; j < N; j++)
            buff_A_out[i][j] = buff_A_out[i][j] + buff_x[i] * buff_y[j];

    lpwr_1: for (i = 0; i < N; i++){
        lpwr_2: for (j = 0; j < N; j++){
            A_out[i][j] =  buff_A_out[i][j];
        }
    }
}