#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <map>
#include <string>
#include <bitset>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <memory>
#include <utility>

using namespace std;

#define BW 32

//#################################################################
float compute_hamm_dist(int curr, int prev);
float average_hamm_dist(int cnt, float avg, float dist);

//#################################################################
class rtl_operator {
public:
    rtl_operator(int op_id);
    bool init(int opnd_num, int op_0, int op_1, int op_2);
    bool update(int opnd_num, int op_0, int op_1, int op_2);

    const int rtlop_id;
    int count;
    vector<float> hamm_dist;
    vector<int> prev_val;
};

//#################################################################
class rtlop_tracer {
public:
    rtlop_tracer(){ };
    ~rtlop_tracer();
    void init_ofstream(const string file_name);
    void trace(int rtlop_id, int opnd_num, int op_0, int op_1, int op_2);
    void print();
    
    static std::unique_ptr<rtlop_tracer> tracer_pt;
    map<int, rtl_operator> rtlop_map;
    ofstream* fout;
};
