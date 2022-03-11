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
#include "rtlop_tracer.h"

using namespace std;

//#################################################################
unique_ptr<rtlop_tracer> build_tracer(string file_name);
rtlop_tracer& get_tracer_pt();