# Benchmark
Here shows the benchmarks described in our paper. 

- target: the target Polybench datasets for evaluation, with full instruction on how to generate the samples
- synthetic: micro-benchmarks that are algebra arithmetics different from the target set, which increases the sample diversity
- utils: some useful toolkits to help with the automatic generation of hardware designs with Vivado and VivadoHLS

## Execution order

Run in ubuntu 16.04 and use atax as an example.

```
Notes:  * Before execution, first need to activate Vivado environment: 

                source <vivado_installation_directory>/settings64.sh

        * Replace the macro PRJ_DIR in ./target/<benchmark>/vivado/src/io_p<x>/rtl/macro.v with the full directory path
```

- Generate a set of HLS designs with the help of files in the folder `./target/atax/hls/auto_run`:

        cd ./target/atax/hls/auto_run
        python auto_run.py atax --src_dir ../src
        cd ../prj
        vivado_hls -f script_0.tcl


- Extract key information and compress the HLS designs using: `utils/hls/compress`:

        cd ./utils/hls/compress
        python compress.py atax --in_dir ../../../target/atax/hls/prj --out_dir ../../../target/atax/hls

- Generate a set of Vivado designs using `utils/vivado/auto_run`:

        cd ./utils/vivado/auto_run
        python auto_run.py --hls_dir ../../../target/atax/hls/prj --src_dir ../../../target/atax/vivado/src --prj_dir ../../../target/atax/vivado/prj
        cd ../../../target/atax/vivado/auto_run
        vivado -mode batch -nojournal -nolog -notrace -source script_0.tcl

- Collect key information from Vivado designs using `utils/vivado/info_extract`:

        cd ./utils/vivado/info_extract
        python info_extract.py --prj_dir ../../../target/atax/vivado/prj
        cd ../../../target/atax/vivado/info_ext
        vivado -mode batch -nojournal -nolog -notrace -source script_0.tcl

- Extract key information and compress the Vivado designs using: `utils/vivado/compress`:

        cd ./utils/vivado/compress
        python compress.py --prj_dir ../../../target/atax/vivado/prj

- We now get the compressed FPGA (bitstream.bit and status_probe.ltx) in folder `./target/atax/vivado/compressed`. Run onboard and measure power directly:

    Refer to .tcl scripts under the folder utils/onboard, to program the bitstreams onboard and set different hardware status

    Finally, store the power with the format in `../../target/atax/power_measurement.csv`

    FPGA board: ZCU102

    Power measurement tool: Zynq UltraScale+ MPSoC Power Advantage Tool 2018.1, check: https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/18841803/Zynq+UltraScale+MPSoC+Power+Advantage+Tool+2018.1?f=print
    
## Required toolkits and versions

   Vivado & VivadoHLS: 2018.2