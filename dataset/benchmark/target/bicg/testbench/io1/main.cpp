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

extern "C" void bicg(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N]);

void bicg_verify(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N]);

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

	DATA_TYPE A[N][N];
	DATA_TYPE p[N];
	DATA_TYPE r[N];
	DATA_TYPE s_out[N];
	DATA_TYPE q_out[N];

	DATA_TYPE golden_A[N][N];
	DATA_TYPE golden_p[N];
	DATA_TYPE golden_r[N];
	DATA_TYPE golden_s_out[N];
	DATA_TYPE golden_q_out[N];
	
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

	sprintf(path, "%s/p_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_p[i] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
			golden_p[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_p[i]);
	    }
    }
	fclose(wptr);

	sprintf(path, "%s/r_in.csv", argv[1]);
	wptr = fopen(path, "wb+");
    for(inv = 0; inv < INV_NUM; inv++){
        for (i = 0; i < N; i++){
			golden_r[i] = ((float)rand() / (float)RAND_MAX) - 0.5 ;  //[-0.5,0.5], signed
			golden_r[i] *= INPUT_SCALE;
			fprintf(wptr, "%.8f\n", golden_r[i]);
	    }
    }
	fclose(wptr);

	FILE *fptr1, *fptr2, *fptr3, *wptr1, *wptr2;
	sprintf(path, "%s/A_in.csv", argv[1]);
	fptr1 = fopen(path, "r");
	sprintf(path, "%s/p_in.csv", argv[1]);
	fptr2 = fopen(path, "r");
	sprintf(path, "%s/r_in.csv", argv[1]);
	fptr3 = fopen(path, "r");
	sprintf(path, "%s/golden_s_out.csv", argv[1]);
    wptr1 = fopen(path, "wb+");
	sprintf(path, "%s/golden_q_out.csv", argv[1]);
    wptr2 = fopen(path, "wb+");

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
			}
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr2, "%f", &read_din) > 0){
        	    golden_p[i] = read_din;
				p[i] = read_din;
			}
			else printf("p: error fscanf to the end!\n");
		}

		for(i = 0; i < N; i++){
			if(fscanf(fptr3, "%f", &read_din) > 0){
        	    golden_r[i] = read_din;
				r[i] = read_din;
			}
			else printf("r: error fscanf to the end!\n");
		}

		//==============================
		bicg(A, p, r, s_out, q_out);
		bicg_verify(golden_A, golden_p, golden_r, golden_s_out, golden_q_out);

		for(i = 0; i < N; i++){
			fprintf(wptr1, "%.4f\n", golden_s_out[i]);
			fprintf(wptr2, "%.4f\n", golden_q_out[i]);
		}
	}

	rtlop_tracer::tracer_pt->print();

	//#################################
    fclose(fptr1);
	fclose(fptr2);
	fclose(fptr3);
	fclose(wptr1);
    fclose(wptr2);
	return 1;
}

//###############################################
void bicg_verify(DATA_TYPE A[N][N], DATA_TYPE p[N], DATA_TYPE r[N], DATA_TYPE s_out[N], DATA_TYPE q_out[N])
{
    int i, j;

    for(i = 0; i < N; i++)
        s_out[i] = SCALAR_VAL(0.0);

    for(i = 0; i < N; i++){
        q_out[i] = SCALAR_VAL(0.0);
        for(j = 0; j < N; j++){
	        s_out[j] = s_out[j] + r[i] * A[i][j];
	        q_out[i] = q_out[i] + A[i][j] * p[j];
	    }
    }
}

