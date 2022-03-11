rm -rf act_trace/build
mkdir act_trace/build
/usr/bin/c++ -fopenmp -O3  -fPIC -std=c++11 -o act_trace/build/rtlop_tracer.o -c act_trace/src/rtlop_tracer.cpp
/usr/bin/c++ -fopenmp -O3  -fPIC -std=c++11 -o act_trace/build/tracer.o -c act_trace/src/tracer.cpp