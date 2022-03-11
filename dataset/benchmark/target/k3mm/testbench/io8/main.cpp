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
#define M 64
#define N 64
#define DATA_TYPE float
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x,y) powf(x,y)

extern "C" void k3mm(DATA_TYPE A_0[N*N/8], DATA_TYPE A_1[N*N/8], DATA_TYPE A_2[N*N/8], DATA_TYPE A_3[N*N/8], 
	DATA_TYPE A_4[N*N/8], DATA_TYPE A_5[N*N/8], DATA_TYPE A_6[N*N/8], DATA_TYPE A_7[N*N/8], 
	DATA_TYPE B_0[N*N/8], DATA_TYPE B_1[N*N/8], DATA_TYPE B_2[N*N/8], DATA_TYPE B_3[N*N/8], 
	DATA_TYPE B_4[N*N/8], DATA_TYPE B_5[N*N/8], DATA_TYPE B_6[N*N/8], DATA_TYPE B_7[N*N/8], 
	DATA_TYPE C_0[N*N/8], DATA_TYPE C_1[N*N/8], DATA_TYPE C_2[N*N/8], DATA_TYPE C_3[N*N/8], 
	DATA_TYPE C_4[N*N/8], DATA_TYPE C_5[N*N/8], DATA_TYPE C_6[N*N/8], DATA_TYPE C_7[N*N/8], 
	DATA_TYPE D_0[N*N/8], DATA_TYPE D_1[N*N/8], DATA_TYPE D_2[N*N/8], DATA_TYPE D_3[N*N/8], 
	DATA_TYPE D_4[N*N/8], DATA_TYPE D_5[N*N/8], DATA_TYPE D_6[N*N/8], DATA_TYPE D_7[N*N/8], 
	DATA_TYPE E_out_0[N*N/8], DATA_TYPE E_out_1[N*N/8], DATA_TYPE E_out_2[N*N/8], DATA_TYPE E_out_3[N*N/8],
	DATA_TYPE E_out_4[N*N/8], DATA_TYPE E_out_5[N*N/8], DATA_TYPE E_out_6[N*N/8], DATA_TYPE E_out_7[N*N/8]);

void k3mm_verify(DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C[M][N], DATA_TYPE D[M][N],DATA_TYPE E_out[M][N]);

//########################################################
int main(int argc, char* argv[])
{
	if (argc < 2)
    {
        cout << "Arguements are not complete. (arg num = " << argc << " < 2)" << "\n";
        return 0;
    }

	//#################################
	srand(123);
	int i, j, inv;
	char path[1024];
	float read_din;

	DATA_TYPE A_0[N*N/8], A_1[N*N/8], A_2[N*N/8], A_3[N*N/8], A_4[N*N/8], A_5[N*N/8], A_6[N*N/8], A_7[N*N/8];
	DATA_TYPE B_0[N*N/8], B_1[N*N/8], B_2[N*N/8], B_3[N*N/8], B_4[N*N/8], B_5[N*N/8], B_6[N*N/8], B_7[N*N/8];
	DATA_TYPE C_0[N*N/8], C_1[N*N/8], C_2[N*N/8], C_3[N*N/8], C_4[N*N/8], C_5[N*N/8], C_6[N*N/8], C_7[N*N/8];
	DATA_TYPE D_0[N*N/8], D_1[N*N/8], D_2[N*N/8], D_3[N*N/8], D_4[N*N/8], D_5[N*N/8], D_6[N*N/8], D_7[N*N/8];
	DATA_TYPE E_out_0[N*N/8], E_out_1[N*N/8], E_out_2[N*N/8], E_out_3[N*N/8], E_out_4[N*N/8], E_out_5[N*N/8], E_out_6[N*N/8], E_out_7[N*N/8];
	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_B[N][N];
    DATA_TYPE golden_C[N][N];
	DATA_TYPE golden_D[N][N];
	DATA_TYPE golden_E_out[N][N];
	
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
				golden_A[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
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
				golden_B[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_B[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_B[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/C_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
		    for (j = 0; j < N; j++){
				golden_C[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_C[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_C[i][j]);
		    }
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/D_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
		    for (j = 0; j < N; j++){
				golden_D[i][j] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
				golden_D[i][j] *= INPUT_SCALE;
			    fprintf(wptr, "%.8f\n", golden_D[i][j]);
		    }
	    }
    }
	fclose(wptr);

	FILE *fptr1, *fptr2, *fptr3, *fptr4, *wptr1;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/B_in.csv", argv[1]);
	fptr2 = fopen(path, "r");
	sprintf(path, "%s/C_in.csv", argv[1]);
	fptr3 = fopen(path, "r");
	sprintf(path, "%s/D_in.csv", argv[1]);
	fptr4 = fopen(path, "r");
	sprintf(path, "%s/golden_E_out.csv", argv[1]);
	wptr1 = fopen(path, "wb+");

	//#################################
    for(inv = 0; inv < INV_NUM; inv++){
        //==============================
		int A_j_0 = 0, A_j_1 = 0, A_j_2 = 0, A_j_3 = 0, A_j_4 = 0, A_j_5 = 0, A_j_6 = 0, A_j_7 = 0;
		int B_j_0 = 0, B_j_1 = 0, B_j_2 = 0, B_j_3 = 0, B_j_4 = 0, B_j_5 = 0, B_j_6 = 0, B_j_7 = 0;
		int C_j_0 = 0, C_j_1 = 0, C_j_2 = 0, C_j_3 = 0, C_j_4 = 0, C_j_5 = 0, C_j_6 = 0, C_j_7 = 0;
		int D_j_0 = 0, D_j_1 = 0, D_j_2 = 0, D_j_3 = 0, D_j_4 = 0, D_j_5 = 0, D_j_6 = 0, D_j_7 = 0;

		for(i = 0; i < N; i++){
		    for(j = 0; j < N; j++){
			    if(fscanf(fptr1, "%f", &read_din) > 0){
        	        golden_A[i][j] = read_din;
					if(j%8 == 0) A_0[A_j_0++] = read_din;
					else if(j%8 == 1) A_1[A_j_1++] = read_din;
					else if(j%8 == 2) A_2[A_j_2++] = read_din;
					else if(j%8 == 3) A_3[A_j_3++] = read_din;
					else if(j%8 == 4) A_4[A_j_4++] = read_din;
					else if(j%8 == 5) A_5[A_j_5++] = read_din;
					else if(j%8 == 6) A_6[A_j_6++] = read_din;
					else if(j%8 == 7) A_7[A_j_7++] = read_din;
				}
				else printf("A: error fscanf to the end!\n");

				if(fscanf(fptr2, "%f", &read_din) > 0){
        	        golden_B[i][j] = read_din;
					if(j%8 == 0) B_0[B_j_0++] = read_din;
					else if(j%8 == 1) B_1[B_j_1++] = read_din;
					else if(j%8 == 2) B_2[B_j_2++] = read_din;
					else if(j%8 == 3) B_3[B_j_3++] = read_din;
					else if(j%8 == 4) B_4[B_j_4++] = read_din;
					else if(j%8 == 5) B_5[B_j_5++] = read_din;
					else if(j%8 == 6) B_6[B_j_6++] = read_din;
					else if(j%8 == 7) B_7[B_j_7++] = read_din;
				}
				else printf("B: error fscanf to the end!\n");

				if(fscanf(fptr3, "%f", &read_din) > 0){
        	        golden_C[i][j] = read_din;
					if(j%8 == 0) C_0[C_j_0++] = read_din;
					else if(j%8 == 1) C_1[C_j_1++] = read_din;
					else if(j%8 == 2) C_2[C_j_2++] = read_din;
					else if(j%8 == 3) C_3[C_j_3++] = read_din;
					else if(j%8 == 4) C_4[C_j_4++] = read_din;
					else if(j%8 == 5) C_5[C_j_5++] = read_din;
					else if(j%8 == 6) C_6[C_j_6++] = read_din;
					else if(j%8 == 7) C_7[C_j_7++] = read_din;
				}
				else printf("C: error fscanf to the end!\n");

				if(fscanf(fptr4, "%f", &read_din) > 0){
        	        golden_D[i][j] = read_din;
					if(j%8 == 0) D_0[D_j_0++] = read_din;
					else if(j%8 == 1) D_1[D_j_1++] = read_din;
					else if(j%8 == 2) D_2[D_j_2++] = read_din;
					else if(j%8 == 3) D_3[D_j_3++] = read_din;
					else if(j%8 == 4) D_4[D_j_4++] = read_din;
					else if(j%8 == 5) D_5[D_j_5++] = read_din;
					else if(j%8 == 6) D_6[D_j_6++] = read_din;
					else if(j%8 == 7) D_7[D_j_7++] = read_din;
				}
				else printf("D: error fscanf to the end!\n");
			}
		}

		//==============================
		k3mm(A_0, A_1, A_2, A_3, A_4, A_5, A_6, A_7, 
			B_0, B_1, B_2, B_3, B_4, B_5, B_6, B_7, 
			C_0, C_1, C_2, C_3, C_4, C_5, C_6, C_7, 
			D_0, D_1, D_2, D_3, D_4, D_5, D_6, D_7, 
			E_out_0, E_out_1, E_out_2, E_out_3, E_out_4, E_out_5, E_out_6, E_out_7);
		k3mm_verify(golden_A, golden_B, golden_C, golden_D,golden_E_out);

		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				fprintf(wptr1, "%.4f\n", golden_E_out[i][j]);
			}
		}
	}

	rtlop_tracer::tracer_pt->print();

	//#################################
    fclose(fptr1);
	fclose(fptr2);
	fclose(fptr3);
	fclose(fptr4);
    fclose(wptr1);
	return 1;
}

//###############################################
void k3mm_verify(DATA_TYPE A[N][N], DATA_TYPE B[N][N], DATA_TYPE C[N][N], DATA_TYPE D[N][N], DATA_TYPE E_out[N][N])
{
    int i, j, k;
	DATA_TYPE E[N][N];
	DATA_TYPE F[N][N];

    /* E := A*B */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            E[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < N; ++k)
                E[i][j] += A[i][k] * B[k][j];
        }
    
    /* F := C*D */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            F[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < N; ++k)
                F[i][j] += C[i][k] * D[k][j];
        }
    
    /* G := E*F */
    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        {
            E_out[i][j] = SCALAR_VAL(0.0);
            for (k = 0; k < N; ++k)
                E_out[i][j] += E[i][k] * F[k][j];
        }
}

