#include "mvm.h"

void mvm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE u1[N], DATA_TYPE v1[N], DATA_TYPE u2[N], DATA_TYPE v2[N],
    DATA_TYPE y[N], DATA_TYPE z[N], DATA_TYPE A_out[N][N], DATA_TYPE x_out[N], DATA_TYPE w_out[N])
{
    int i, j;

    DATA_TYPE buff_A[N][N];
    DATA_TYPE buff_u1[N];
    DATA_TYPE buff_v1[N];
    DATA_TYPE buff_u2[N];
    DATA_TYPE buff_v2[N];
    DATA_TYPE buff_y[N];
    DATA_TYPE buff_z[N];
    DATA_TYPE buff_w_out[N];
    DATA_TYPE buff_x_out[N];
    
    lprd_1: for (i = 0; i < N; i++){
        buff_u1[i] = u1[i];
        buff_v1[i] = v1[i];
        buff_u2[i] = u2[i];
        buff_v2[i] = v2[i];
        buff_y[i] = y[i];
        buff_z[i] = z[i];
        buff_x_out[i] = 0;
        buff_w_out[i] = 0;
        lprd_2: for (j = 0; j < N; j++){
            buff_A[i][j] = A[i][j];
        }
    }

    lp1: for (i = 0; i < N; i++)
        lp2: for (j = 0; j < N; j++)
            buff_A[i][j] += buff_u1[i] * buff_v1[j] + buff_u2[i] * buff_v2[j];

    lp3: for (i = 0; i < N; i++)
        lp4: for (j = 0; j < N; j++)
            buff_x_out[i] += beta * buff_A[j][i] * buff_y[j];

    lp5: for (i = 0; i < N; i++)
        buff_x_out[i] = buff_x_out[i] + buff_z[i];

    lp6: for (i = 0; i < N; i++)
        lp7: for (j = 0; j < N; j++)
            buff_w_out[i] += alpha * buff_A[i][j] * buff_x_out[j];

    lpwr_1: for (i = 0; i < N; i++){
        w_out[i] = buff_w_out[i];
        x_out[i] = buff_x_out[i];
        lpwr_2: for (j = 0; j < N; j++){
            A_out[i][j] = buff_A[i][j];
        }
    }
}
