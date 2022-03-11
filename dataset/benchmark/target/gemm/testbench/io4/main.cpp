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

extern "C" void gemm(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A_0[N*N/4], DATA_TYPE A_1[N*N/4], DATA_TYPE A_2[N*N/4], DATA_TYPE A_3[N*N/4], 
	DATA_TYPE B_0[N*N/4], DATA_TYPE B_1[N*N/4], DATA_TYPE B_2[N*N/4], DATA_TYPE B_3[N*N/4], 
	DATA_TYPE C_0[N*N/4], DATA_TYPE C_1[N*N/4], DATA_TYPE C_2[N*N/4], DATA_TYPE C_3[N*N/4], 
	DATA_TYPE D_out_0[N*N/4], DATA_TYPE D_out_1[N*N/4], DATA_TYPE D_out_2[N*N/4], DATA_TYPE D_out_3[N*N/4]);

void gemm_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N]);

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

	DATA_TYPE alpha = 0.25;
	DATA_TYPE beta = -0.333;
	DATA_TYPE A_0[N*N/4], A_1[N*N/4], A_2[N*N/4], A_3[N*N/4];
	DATA_TYPE B_0[N*N/4], B_1[N*N/4], B_2[N*N/4], B_3[N*N/4];
	DATA_TYPE C_0[N*N/4], C_1[N*N/4], C_2[N*N/4], C_3[N*N/4];
	DATA_TYPE D_out_0[N*N/4], D_out_1[N*N/4], D_out_2[N*N/4], D_out_3[N*N/4];
	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_B[N][N];
    DATA_TYPE golden_C[N][N];
	DATA_TYPE golden_D_out[N][N];
	
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
    for(inv=0; inv<INV_NUM; inv++){
        for (i=0; i<N; i++){
		    for (j=0; j<N; j++){
				golden_A[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_A[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_A[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/B_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv=0; inv<INV_NUM; inv++){
        for (i=0; i<N; i++){
		    for (j=0; j<N; j++){
				golden_B[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_B[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_B[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/C_in.csv", argv[1]);
	wptr = fopen(path,"wb+");
    for(inv=0; inv<INV_NUM; inv++){
        for (i=0; i<N; i++){
		    for (j=0; j<N; j++){
				golden_C[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_C[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_C[i][j]);
		    }
	    }
    }
	fclose(wptr);

	FILE *fptr1, *fptr2, *fptr3, *wptr1;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/B_in.csv", argv[1]);
    fptr2 = fopen(path, "r");
	sprintf(path, "%s/C_in.csv", argv[1]);
    fptr3 = fopen(path, "r");
	sprintf(path, "%s/golden_D_out.csv", argv[1]);
    wptr1 = fopen(path, "wb+");

	//#################################
    for(inv = 0; inv < INV_NUM; inv++){
        //==============================
		int A_j_0 = 0, A_j_1 = 0, A_j_2 = 0, A_j_3 = 0;
		int B_j_0 = 0, B_j_1 = 0, B_j_2 = 0, B_j_3 = 0;
		int C_j_0 = 0, C_j_1 = 0, C_j_2 = 0, C_j_3 = 0;

		for (i = 0; i < N; i++){
		    for (j = 0; j < N; j++){
			    if(fscanf(fptr1, "%f", &read_din) > 0){
        	        golden_A[i][j] = read_din;
					if(j%4 == 0) A_0[A_j_0++] = read_din;
					else if(j%4 == 1) A_1[A_j_1++] = read_din;
					else if(j%4 == 2) A_2[A_j_2++] = read_din;
					else if(j%4 == 3) A_3[A_j_3++] = read_din;
				}
				else printf("A: error fscanf to the end!\n");

				if(fscanf(fptr2, "%f", &read_din) > 0){
        	        golden_B[i][j] = read_din;
					if(j%4 == 0) B_0[B_j_0++] = read_din;
					else if(j%4 == 1) B_1[B_j_1++] = read_din;
					else if(j%4 == 2) B_2[B_j_2++] = read_din;
					else if(j%4 == 3) B_3[B_j_3++] = read_din;
				}
				else printf("B: error fscanf to the end!\n");

				if(fscanf(fptr3, "%f", &read_din) > 0){
        	        golden_C[i][j] = read_din;
					if(j%4 == 0) C_0[C_j_0++] = read_din;
					else if(j%4 == 1) C_1[C_j_1++] = read_din;
					else if(j%4 == 2) C_2[C_j_2++] = read_din;
					else if(j%4 == 3) C_3[C_j_3++] = read_din;
				}
				else printf("C: error fscanf to the end!\n");
			}
		}

		//==============================
		gemm(alpha, beta, A_0, A_1, A_2, A_3, B_0, B_1, B_2, B_3, C_0, C_1, C_2, C_3, D_out_0, D_out_1, D_out_2, D_out_3);
		gemm_verify(alpha, beta, golden_A, golden_B, golden_C, golden_D_out);

		for(i = 0;i < N; i++){
			for(j = 0; j < N; j++){
				fprintf(wptr1, "%.4f\n", golden_D_out[i][j]);
			}
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
void gemm_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D_out[N][N])
{
    int i, j, k;
	DATA_TYPE tmp1[N][N];

    lp1: for (i = 0; i < N; i++) {
        lp2: for (j = 0; j < N; j++) {
			tmp1[i][j] = 0;
            lp3: for (k = 0; k < N; k++) {
                tmp1[i][j] = alpha * A[i][k] * B[k][j] + tmp1[i][j];
            }
        }
    }

    lp4: for (i = 0; i < N; i++) {
        lp5: for (j = 0; j < N; j++) {
	        D_out[i][j] = beta * C[i][j] + tmp1[i][j];
        }
    }
}
