`include "macro.v"
module wrapper(clk_p, clk_n, ap_rst, probe_out, data_out, data_valid);

parameter IO_PARTITION_FACTOR = 4; //change according to different io partition factors
parameter DATASET_UPDATE_INV = 1;  //change to reduce ram writing frequency
parameter INV_DATASET_SIZE = 4096; //in[64][64] = 4096

parameter DATA_SIZE_PER_RAM = (INV_DATASET_SIZE / IO_PARTITION_FACTOR);
parameter DATA_WIDTH = 64;
parameter DATASET_NUM = 8;

//###############################
input clk_p;
input clk_n;
input ap_rst;
output probe_out;
output reg [3:0] data_out;
output reg data_valid;

wire ap_clk;
clk_wiz_0 gen_clk(.clk_in1_p(clk_p), .clk_in1_n(clk_n), .clk_out1(ap_clk), .reset(1'b0), .locked());

wire ap_done;
wire ap_idle;
wire ap_ready;
reg ap_start;

//###############################
wire [9:0] A_0_address0;
wire A_0_ce0;
wire [31:0] A_0_q0;
wire [9:0] A_1_address0;
wire A_1_ce0;
wire [31:0] A_1_q0;
wire [9:0] A_2_address0;
wire A_2_ce0;
wire [31:0] A_2_q0;
wire [9:0] A_3_address0;
wire A_3_ce0;
wire [31:0] A_3_q0;
wire [9:0] B_0_address0;
wire B_0_ce0;
wire [31:0] B_0_q0;
wire [9:0] B_1_address0;
wire B_1_ce0;
wire [31:0] B_1_q0;
wire [9:0] B_2_address0;
wire B_2_ce0;
wire [31:0] B_2_q0;
wire [9:0] B_3_address0;
wire B_3_ce0;
wire [31:0] B_3_q0;
wire [9:0] C_0_address0;
wire C_0_ce0;
wire [31:0] C_0_q0;
wire [9:0] C_1_address0;
wire C_1_ce0;
wire [31:0] C_1_q0;
wire [9:0] C_2_address0;
wire C_2_ce0;
wire [31:0] C_2_q0;
wire [9:0] C_3_address0;
wire C_3_ce0;
wire [31:0] C_3_q0;
wire [9:0] D_0_address0;
wire D_0_ce0;
wire [31:0] D_0_q0;
wire [9:0] D_1_address0;
wire D_1_ce0;
wire [31:0] D_1_q0;
wire [9:0] D_2_address0;
wire D_2_ce0;
wire [31:0] D_2_q0;
wire [9:0] D_3_address0;
wire D_3_ce0;
wire [31:0] D_3_q0;
wire [31:0] E_out_0_din;
wire E_out_0_write;
wire [31:0] E_out_1_din;
wire E_out_1_write;
wire [31:0] E_out_2_din;
wire E_out_2_write;
wire [31:0] E_out_3_din;
wire E_out_3_write;

//###############################
vio_0 vio_inst(.clk(ap_clk), .probe_out0(probe_out));

reg pp_ap_start, p_ap_start;
always@(posedge ap_clk)
begin
    pp_ap_start <= probe_out;
    p_ap_start <= pp_ap_start;
    ap_start <= p_ap_start;
end

//############# A #############
kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/A_rom_0.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/A_ram_0_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/A_ram_0_1.mif"}))
A_ram_0 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(A_0_ce0), .kram_addr(A_0_address0), .kram_dout(A_0_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/A_rom_1.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/A_ram_1_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/A_ram_1_1.mif"}))
A_ram_1 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(A_1_ce0), .kram_addr(A_1_address0), .kram_dout(A_1_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/A_rom_2.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/A_ram_2_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/A_ram_2_1.mif"}))
A_ram_2 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(A_2_ce0), .kram_addr(A_2_address0), .kram_dout(A_2_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/A_rom_3.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/A_ram_3_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/A_ram_3_1.mif"}))
A_ram_3 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(A_3_ce0), .kram_addr(A_3_address0), .kram_dout(A_3_q0));

//############# B #############
kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/B_rom_0.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/B_ram_0_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/B_ram_0_1.mif"}))
B_ram_0 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(B_0_ce0), .kram_addr(B_0_address0), .kram_dout(B_0_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/B_rom_1.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/B_ram_1_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/B_ram_1_1.mif"}))
B_ram_1 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(B_1_ce0), .kram_addr(B_1_address0), .kram_dout(B_1_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/B_rom_2.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/B_ram_2_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/B_ram_2_1.mif"}))
B_ram_2 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(B_2_ce0), .kram_addr(B_2_address0), .kram_dout(B_2_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/B_rom_3.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/B_ram_3_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/B_ram_3_1.mif"}))
B_ram_3 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(B_3_ce0), .kram_addr(B_3_address0), .kram_dout(B_3_q0));

//############# C #############
kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/C_rom_0.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/C_ram_0_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/C_ram_0_1.mif"}))
C_ram_0 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(C_0_ce0), .kram_addr(C_0_address0), .kram_dout(C_0_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/C_rom_1.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/C_ram_1_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/C_ram_1_1.mif"}))
C_ram_1 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(C_1_ce0), .kram_addr(C_1_address0), .kram_dout(C_1_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/C_rom_2.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/C_ram_2_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/C_ram_2_1.mif"}))
C_ram_2 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(C_2_ce0), .kram_addr(C_2_address0), .kram_dout(C_2_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/C_rom_3.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/C_ram_3_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/C_ram_3_1.mif"}))
C_ram_3 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(C_3_ce0), .kram_addr(C_3_address0), .kram_dout(C_3_q0));

//############# D #############
kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/D_rom_0.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/D_ram_0_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/D_ram_0_1.mif"}))
D_ram_0 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(D_0_ce0), .kram_addr(D_0_address0), .kram_dout(D_0_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/D_rom_1.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/D_ram_1_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/D_ram_1_1.mif"}))
D_ram_1 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(D_1_ce0), .kram_addr(D_1_address0), .kram_dout(D_1_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/D_rom_2.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/D_ram_2_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/D_ram_2_1.mif"}))
D_ram_2 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(D_2_ce0), .kram_addr(D_2_address0), .kram_dout(D_2_q0));

kernel_ram #(.DATA_WIDTH(DATA_WIDTH), .DATA_SIZE(DATA_SIZE_PER_RAM), .RAM_ADDR_WIDTH(`CLOG2(DATA_SIZE_PER_RAM)), .RAM_UPDATE_INV(DATASET_UPDATE_INV),
    .DATASET_NUM(DATASET_NUM), .ROM_ADDR_WIDTH((`CLOG2(DATA_SIZE_PER_RAM*DATASET_NUM))), 
    .ROM_INIT_FILE({`PRJ_DIR, "mif/D_rom_3.mif"}), 
    .RAM_INIT_FILE_0({`PRJ_DIR, "mif/D_ram_3_0.mif"}), 
    .RAM_INIT_FILE_1({`PRJ_DIR, "mif/D_ram_3_1.mif"}))
D_ram_3 (.ap_clk(ap_clk), .ap_rst(ap_rst), .ap_start(ap_start), .ap_done(ap_done), 
    .kram_en(D_3_ce0), .kram_addr(D_3_address0), .kram_dout(D_3_q0));

//###############################
k3mm kernel(
	.ap_clk(ap_clk),
	.ap_rst(ap_rst),
	.ap_start(ap_start),
	.ap_done(ap_done),
	.ap_idle(ap_idle),
	.ap_ready(ap_ready),
	.A_0_address0(A_0_address0),
	.A_0_ce0(A_0_ce0),
	.A_0_q0(A_0_q0),
	.A_1_address0(A_1_address0),
	.A_1_ce0(A_1_ce0),
	.A_1_q0(A_1_q0),
	.A_2_address0(A_2_address0),
	.A_2_ce0(A_2_ce0),
	.A_2_q0(A_2_q0),
	.A_3_address0(A_3_address0),
	.A_3_ce0(A_3_ce0),
	.A_3_q0(A_3_q0),
	.B_0_address0(B_0_address0),
	.B_0_ce0(B_0_ce0),
	.B_0_q0(B_0_q0),
	.B_1_address0(B_1_address0),
	.B_1_ce0(B_1_ce0),
	.B_1_q0(B_1_q0),
	.B_2_address0(B_2_address0),
	.B_2_ce0(B_2_ce0),
	.B_2_q0(B_2_q0),
	.B_3_address0(B_3_address0),
	.B_3_ce0(B_3_ce0),
	.B_3_q0(B_3_q0),
	.C_0_address0(C_0_address0),
	.C_0_ce0(C_0_ce0),
	.C_0_q0(C_0_q0),
	.C_1_address0(C_1_address0),
	.C_1_ce0(C_1_ce0),
	.C_1_q0(C_1_q0),
	.C_2_address0(C_2_address0),
	.C_2_ce0(C_2_ce0),
	.C_2_q0(C_2_q0),
	.C_3_address0(C_3_address0),
	.C_3_ce0(C_3_ce0),
	.C_3_q0(C_3_q0),
    .D_0_address0(D_0_address0),
	.D_0_ce0(D_0_ce0),
	.D_0_q0(D_0_q0),
	.D_1_address0(D_1_address0),
	.D_1_ce0(D_1_ce0),
	.D_1_q0(D_1_q0),
	.D_2_address0(D_2_address0),
	.D_2_ce0(D_2_ce0),
	.D_2_q0(D_2_q0),
	.D_3_address0(D_3_address0),
	.D_3_ce0(D_3_ce0),
	.D_3_q0(D_3_q0),
	.E_out_0_din(E_out_0_din),
	.E_out_0_full_n(1'b1),
	.E_out_0_write(E_out_0_write),
	.E_out_1_din(E_out_1_din),
	.E_out_1_full_n(1'b1),
	.E_out_1_write(E_out_1_write),
	.E_out_2_din(E_out_2_din),
	.E_out_2_full_n(1'b1),
	.E_out_2_write(E_out_2_write),
	.E_out_3_din(E_out_3_din),
	.E_out_3_full_n(1'b1),
	.E_out_3_write(E_out_3_write)
);

//############## output stage 1: xor output by itself ##############
reg [8-1:0] xor_1 [3:0];
reg xor_valid_1 [3:0];

always@(posedge ap_clk)
begin
    xor_valid_1[0] <= E_out_0_write;
	xor_valid_1[1] <= E_out_1_write;
    xor_valid_1[2] <= E_out_2_write;
	xor_valid_1[3] <= E_out_3_write;

    xor_1[0] <= E_out_0_din[7:0] ^ E_out_0_din[15:8] ^ E_out_0_din[23:16] ^ E_out_0_din[31:24];
	xor_1[1] <= E_out_1_din[7:0] ^ E_out_1_din[15:8] ^ E_out_1_din[23:16] ^ E_out_1_din[31:24];
    xor_1[2] <= E_out_2_din[7:0] ^ E_out_2_din[15:8] ^ E_out_2_din[23:16] ^ E_out_2_din[31:24];
	xor_1[3] <= E_out_3_din[7:0] ^ E_out_3_din[15:8] ^ E_out_3_din[23:16] ^ E_out_3_din[31:24];
end

//############## output stage 2: 2-to-1 reduction: select/xor ##############
reg [8-1:0] xor_2 [1:0];
reg [1:0] xor_valid_2;

always@(posedge ap_clk)
begin
    xor_valid_2[0] <= xor_valid_1[0] | xor_valid_1[1];
    xor_valid_2[1] <= xor_valid_1[2] | xor_valid_1[3];
    
    case({xor_valid_1[1], xor_valid_1[0]})
        2'b00: xor_2[0] <= 0;
        2'b01: xor_2[0] <= xor_1[0];
        2'b10: xor_2[0] <= xor_1[1];
        2'b11: xor_2[0] <= xor_1[0] ^ xor_1[1];
    endcase

    case({xor_valid_1[3], xor_valid_1[2]})
        2'b00: xor_2[1] <= 0;
        2'b01: xor_2[1] <= xor_1[2];
        2'b10: xor_2[1] <= xor_1[3];
        2'b11: xor_2[1] <= xor_1[2] ^ xor_1[3];
    endcase
end

//############## output stage 3: 2-to-1 reduction: select/xor ##############
reg [8-1:0] xor_3;
reg xor_valid_3;

always@(posedge ap_clk)
begin
    xor_valid_3 <= xor_valid_2[0] | xor_valid_2[1];
    
    case({xor_valid_2[1], xor_valid_2[0]})
        2'b00: xor_3 <= 0;
        2'b01: xor_3 <= xor_2[0];
        2'b10: xor_3 <= xor_2[1];
        2'b11: xor_3 <= xor_2[0] ^ xor_2[1];
    endcase
end

always@(posedge ap_clk)
begin
    data_valid <= xor_valid_3;
    data_out <= xor_3[7:4] ^ xor_3[3:0];
end

endmodule