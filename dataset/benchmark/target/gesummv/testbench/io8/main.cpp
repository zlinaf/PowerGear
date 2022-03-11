/* #####################################
  argv[1]: input_path
  argv[2]: node / edge
#####################################*/

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <math.h>
#include <memory>
#include <utility>
#include "tracer.h"

using namespace std;

#define INV_NUM 8
#define INPUT_SCALE 1
#define N 64
#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

extern "C" void gesummv(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A_0[N*N/8], DATA_TYPE A_1[N*N/8], DATA_TYPE A_2[N*N/8], DATA_TYPE A_3[N*N/8], 
	DATA_TYPE A_4[N*N/8], DATA_TYPE A_5[N*N/8], DATA_TYPE A_6[N*N/8], DATA_TYPE A_7[N*N/8], 
	DATA_TYPE B_0[N*N/8], DATA_TYPE B_1[N*N/8], DATA_TYPE B_2[N*N/8], DATA_TYPE B_3[N*N/8], 
	DATA_TYPE B_4[N*N/8], DATA_TYPE B_5[N*N/8], DATA_TYPE B_6[N*N/8], DATA_TYPE B_7[N*N/8], 
	DATA_TYPE x[N], DATA_TYPE y_out[N]);

void gesummv_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N]);

//########################################################
int main(int argc, char* argv[])
{
	if (argc < 2)
    {
        cout << "Arguements are not complete. (arg num = " << argc << " < 2)" << "\n";
        return 0;
    }

	//#################################
    srand(123); //hls test seed is 123
	int i, j, inv;
	char path[1024];
	float read_din;

	DATA_TYPE alpha = -0.125;
	DATA_TYPE beta = 0.333;
	DATA_TYPE A_0[N*N/8], A_1[N*N/8], A_2[N*N/8], A_3[N*N/8];
	DATA_TYPE A_4[N*N/8], A_5[N*N/8], A_6[N*N/8], A_7[N*N/8];
	DATA_TYPE B_0[N*N/8], B_1[N*N/8], B_2[N*N/8], B_3[N*N/8];
	DATA_TYPE B_4[N*N/8], B_5[N*N/8], B_6[N*N/8], B_7[N*N/8];
	DATA_TYPE x[N];
	DATA_TYPE y_out[N];
	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_B[N][N];
	DATA_TYPE golden_x[N];
	DATA_TYPE golden_y_out[N];
	
	sprintf(path, "%s/hd_%s.csv", argv[1], argv[2]);
	rtlop_tracer::tracer_pt = std::move(build_tracer(path));
	if(rtlop_tracer::tracer_pt == nullptr)
	{
		cout << "rtlop_tracer::tracer_pt = nullptr." << "\n";
        return 0;
	}

    //input stimulus generation, can only generate once
	FILE *wptr;
	sprintf(path, "%s/A_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
		    for (j = 0; j < N; j++){
				golden_A[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
				golden_A[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_A[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/B_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
		    for (j = 0; j < N; j++){
				golden_B[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
				golden_B[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_B[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/x_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_x[i] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
			golden_x[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_x[i]);
	    }
    }
	fclose(wptr);

	FILE *fptr1, *fptr2, *fptr3, *wptr1;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/B_in.csv", argv[1]);
	fptr2 = fopen(path, "r");
	sprintf(path, "%s/x_in.csv", argv[1]);
	fptr3 = fopen(path, "r");
	sprintf(path, "%s/golden_y_out.csv", argv[1]);
    wptr1 = fopen(path, "wb+");

	//#################################
    for(inv = 0; inv < INV_NUM; inv++){
        //==============================
		int j_A_0 = 0, j_A_1 = 0, j_A_2 = 0, j_A_3 = 0;
		int j_A_4 = 0, j_A_5 = 0, j_A_6 = 0, j_A_7 = 0;
		int j_B_0 = 0, j_B_1 = 0, j_B_2 = 0, j_B_3 = 0;
		int j_B_4 = 0, j_B_5 = 0, j_B_6 = 0, j_B_7 = 0;

		for(i = 0; i < N; i++){
		    for(j = 0; j < N; j++){
			    if(fscanf(fptr1, "%f", &read_din) > 0){
        	        golden_A[i][j] = read_din;
					if(j%8 == 0) A_0[j_A_0++] = read_din;
					else if(j%8 == 1) A_1[j_A_1++] = read_din;
					else if(j%8 == 2) A_2[j_A_2++] = read_din;
					else if(j%8 == 3) A_3[j_A_3++] = read_din;
					else if(j%8 == 4) A_4[j_A_4++] = read_din;
					else if(j%8 == 5) A_5[j_A_5++] = read_din;
					else if(j%8 == 6) A_6[j_A_6++] = read_din;
					else A_7[j_A_7++] = read_din;
				}
				else printf("A: error fscanf to the end!\n");
			}
		}

		for(i = 0; i < N; i++){
		    for(j = 0; j < N; j++){
			    if(fscanf(fptr2, "%f", &read_din) > 0){
        	        golden_B[i][j] = read_din;
					if(j%8 == 0) B_0[j_B_0++] = read_din;
					else if(j%8 == 1) B_1[j_B_1++] = read_din;
					else if(j%8 == 2) B_2[j_B_2++] = read_din;
					else if(j%8 == 3) B_3[j_B_3++] = read_din;
					else if(j%8 == 4) B_4[j_B_4++] = read_din;
					else if(j%8 == 5) B_5[j_B_5++] = read_din;
					else if(j%8 == 6) B_6[j_B_6++] = read_din;
					else B_7[j_B_7++] = read_din;
				}
				else printf("B: error fscanf to the end!\n");
			}
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr3, "%f", &read_din) > 0){
        	    golden_x[i] = read_din;
				x[i] = read_din;
			}
			else printf("x: error fscanf to the end!\n");
		}

		//==============================
		gesummv(alpha, beta, A_0, A_1, A_2, A_3, A_4, A_5, A_6, A_7, B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, x, y_out);
		gesummv_verify(alpha, beta, golden_A, golden_B, golden_x, golden_y_out);

		for (i = 0; i < N; i++){
			fprintf(wptr1, "%.4f\n", golden_y_out[i]);
		}
	}

	rtlop_tracer::tracer_pt->print();

	//#################################
    fclose(fptr1);
	fclose(fptr2);
	fclose(fptr3);
	fclose(wptr1);
	return 1;
}

//###############################################
void gesummv_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
	int i, j;
	DATA_TYPE tmp1[N];
	DATA_TYPE tmp2[N];
	
    lp1: for(i = 0; i < N; i++) {
		tmp1[i] = 0;
		tmp2[i] = 0;
        lp2: for(j = 0; j < N; j++) {
	        tmp1[i] += alpha * A[i][j] * x[j];
			tmp2[i] += beta * B[i][j] * x[j];
        }
    }

	lp3: for(i = 0; i < N; i++) {
		y_out[i] = tmp1[i] + tmp2[i];
	}
}

/*
void gesummv_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE x[N], DATA_TYPE y_out[N])
{
    int i, j;
    DATA_TYPE tmp[N];

    for (i = 0; i < N; i++){
        tmp[i] = SCALAR_VAL(0.0);
        y_out[i] = SCALAR_VAL(0.0);
        for (j = 0; j < N; j++){
	        tmp[i] = A[i][j] * x[j] + tmp[i];
	        y_out[i] = B[i][j] * x[j] + y_out[i];
        }
        y_out[i] = alpha * tmp[i] + beta * y_out[i];
    }
}
*/
