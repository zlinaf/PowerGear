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

extern "C" void mvt(DATA_TYPE A_0[N*N/2], DATA_TYPE A_1[N*N/2], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N]);

void mvt_verify(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N]);

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

	DATA_TYPE A_0[N*N/2], A_1[N*N/2];
	DATA_TYPE x1[N];
	DATA_TYPE x2[N];
	DATA_TYPE y1[N];
	DATA_TYPE y2[N];
	DATA_TYPE x1_out[N];
	DATA_TYPE x2_out[N];
	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_x1[N];
	DATA_TYPE golden_x2[N];
	DATA_TYPE golden_y1[N];
	DATA_TYPE golden_y2[N];
	DATA_TYPE golden_x1_out[N];
	DATA_TYPE golden_x2_out[N];
	
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

	sprintf(path, "%s/x1_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_x1[i] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
			golden_x1[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_x1[i]);
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/x2_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_x2[i] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
			golden_x2[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_x2[i]);
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/y1_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_y1[i] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
			golden_y1[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_y1[i]);
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/y2_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_y2[i] = ((float)rand() / (float)RAND_MAX) - 0.5; //[-0.5,0.5], signed
			golden_y2[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_y2[i]);
	    }
    }
	fclose(wptr);

	FILE *fptr1, *fptr2, *fptr3, *fptr4, *fptr5, *wptr1, *wptr2;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/x1_in.csv", argv[1]);
	fptr2 = fopen(path, "r");
	sprintf(path, "%s/x2_in.csv", argv[1]);
	fptr3 = fopen(path, "r");
	sprintf(path, "%s/y1_in.csv", argv[1]);
	fptr4 = fopen(path, "r");
	sprintf(path, "%s/y2_in.csv", argv[1]);
	fptr5 = fopen(path, "r");
	sprintf(path, "%s/golden_x1_out.csv", argv[1]);
    wptr1 = fopen(path, "wb+");
	sprintf(path, "%s/golden_x2_out.csv", argv[1]);
    wptr2 = fopen(path, "wb+");

	//#################################
    for(inv = 0; inv < INV_NUM; inv++){
        //==============================
		int j_A_0 = 0, j_A_1 = 0;

		for(i = 0; i < N; i++){
		    for(j = 0; j < N; j++){
			    if(fscanf(fptr1, "%f", &read_din) > 0){
        	        golden_A[i][j] = read_din;
					if(j%2 == 0) A_0[j_A_0++] = read_din;
					else A_1[j_A_1++] = read_din;
				}
				else printf("A: error fscanf to the end!\n");
			}
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr2, "%f", &read_din) > 0){
        	    golden_x1[i] = read_din;
				x1[i] = read_din;
			}
			else printf("x1: error fscanf to the end!\n");
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr3, "%f", &read_din) > 0){
        	    golden_x2[i] = read_din;
				x2[i] = read_din;
			}
			else printf("x2: error fscanf to the end!\n");
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr4, "%f", &read_din) > 0){
        	    golden_y1[i] = read_din;
				y1[i] = read_din;
			}
			else printf("y1: error fscanf to the end!\n");
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr5, "%f", &read_din) > 0){
        	    golden_y2[i] = read_din;
				y2[i] = read_din;
			}
			else printf("y2: error fscanf to the end!\n");
		}

		//==============================
		mvt(A_0, A_1, x1, x2, y1, y2, x1_out, x2_out);
		mvt_verify(golden_A, golden_x1, golden_x2, golden_y1, golden_y2, golden_x1_out, golden_x2_out);

		for(i=0; i<N; i++){
			fprintf(wptr1, "%.4f\n", golden_x1[i]);
			fprintf(wptr2, "%.4f\n", golden_x2[i]);
		}
	}

	rtlop_tracer::tracer_pt->print();

	//#################################
    fclose(fptr1);
	fclose(fptr2);
	fclose(fptr3);
	fclose(fptr4);
	fclose(fptr5);
	fclose(wptr1);
    fclose(wptr2);
	return 1;
}

//###############################################
void mvt_verify(DATA_TYPE A[N][N], DATA_TYPE x1[N], DATA_TYPE x2[N], DATA_TYPE y1[N], DATA_TYPE y2[N], DATA_TYPE x1_out[N], DATA_TYPE x2_out[N])
{
    int i, j;
	for (i = 0; i < N; i++){
		x1_out[i] = x1[i];
		x2_out[i] = x2[i];
	}

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            x1[i] = x1[i] + A[i][j] * y1[j];

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            x2[i] = x2[i] + A[j][i] * y2[j];

	for (i = 0; i < N; i++) {
		x1_out[i] = x1[i];
		x2_out[i] = x2[i];
	}
}

