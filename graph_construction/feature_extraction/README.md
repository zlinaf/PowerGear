# Feature Extraction

This is the automatic feature extraction flow. 
Before running the flow, make sure that the HLS designs are properly collected and compressed (check how to do in the benchmark folder).

## How to use this flow
- Install LLVM 11.0.0 and build the following binaries (only need to build once):
   - To build act_trace: 
      
         sh build_act_trace.sh

   - To build ir_revise: 

         sh build_ir_revise.sh

   - The compiled output files are in `act_trace/build and ir_revise/build`, respectively

- Go to the overall_run folder, try with a simple demo: 
      
      python feature_extract.py atax --app_dir ../../dataset/benchmark/target

   - Check the result in the following folder: `../../dataset/benchmark/target/atax/generated`
         