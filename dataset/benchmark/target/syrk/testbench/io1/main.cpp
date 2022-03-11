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

extern "C" void syrk(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C_out[M][N]);

void syrk_verify(DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE A[M][N], DATA_TYPE B[M][N], DATA_TYPE C_out[M][N]);

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
	DATA_TYPE A[N][N];
	DATA_TYPE B[N][N];
	DATA_TYPE C_out[N][N];
	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_B[N][N];
	DATA_TYPE golden_C_out[N][N];
	DATA_TYPE alpha = -0.125;
    DATA_TYPE beta = 0.333;

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
	
	FILE *fptr1, *fptr2, *wptr1;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/B_in.csv", argv[1]);
	fptr2 = fopen(path, "r");
	sprintf(path, "%s/golden_C_out.csv", argv[1]);
	wptr1 = fopen(path, "wb+");

	//#################################
    for(inv = 0; inv < INV_NUM; inv++){
        //==============================
		for(i = 0; i < N; i++){
		    for(j = 0; j < N; j++){
			    if(fscanf(fptr1, "%f", &read_din) > 0){
        	        golden_A[i][j] = read_din;
					A[i][j] = read_din;
				}
				else printf("A: error fscanf to the end!\n");

				if(fscanf(fptr2, "%f", &read_din) > 0){
        	        golden_B[i][j] = read_din;
					B[i][j] = read_din;
				}
				else printf("B: error fscanf to the end!\n");
			}
		}

		//==============================
		syrk(alpha,beta,A, B,  C_out);
		syrk_verify(alpha,beta,golden_A, golden_B,  golden_C_out);

		for(i = 0; i < N; i++){
			for(j = 0; j < N; j++){
				fprintf(wptr1, "%.4f\n", golden_C_out[i][j]);
			}
		}
	}

	rtlop_tracer::tracer_pt->print();

	//#################################
    fclose(fptr1);
	fclose(fptr2);
    fclose(wptr1);
	return 1;
}

//###############################################
void syrk_verify(DATA_TYPE alpha,DATA_TYPE beta,DATA_TYPE A[N][N], DATA_TYPE B[N][N] ,DATA_TYPE C_out[N][N]) {
	int i, j, k;

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			C_out[i][j] = 0;
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			if (j > i) continue;
			for (k = 0; k < N; k++) {
				C_out[i][j] += alpha * A[i][k] * A[j][k];
			}
		}
	}

	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			if (j > i) continue;
			C_out[i][j] += beta * B[i][j];
		}
	}
}

