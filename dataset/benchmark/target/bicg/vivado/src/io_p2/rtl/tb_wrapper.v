`timescale 1 ns / 1 ps 
`include "macro.v"
module tb;

//################################################
reg clk_p;
wire clk_n;
reg ap_rst;
wire probe_out;
wire [3:0] data_out;
wire data_valid;

wrapper top(.clk_p(clk_p), .clk_n(clk_n), .ap_rst(ap_rst), .probe_out(probe_out), .data_out(data_out), .data_valid(data_valid));

//################################################
initial
begin
    clk_p = 0;
    ap_rst = 1;
    #100 ap_rst = 0;
end

always #1.667 clk_p = ~clk_p;
assign clk_n = ~clk_p;

//################################################
integer rec_s_out_0;
initial rec_s_out_0 = $fopen("fpga_s_out_0.txt","w");
always@(posedge top. ap_clk)
begin
    if(top. s_out_write)
    begin
        $fwrite(rec_s_out_0, "%h\n", top. s_out_din);
    end
end

integer rec_q_out_0;
initial rec_q_out_0 = $fopen("fpga_q_out_0.txt","w");
always@(posedge top. ap_clk)
begin
    if(top. q_out_write)
    begin
        $fwrite(rec_q_out_0, "%h\n", top. q_out_din);
    end
end

endmodule