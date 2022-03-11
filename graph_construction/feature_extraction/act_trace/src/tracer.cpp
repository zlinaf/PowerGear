#include "tracer.h"

//#################################################################
unique_ptr<rtlop_tracer> build_tracer(string file_name)
{
    unique_ptr<rtlop_tracer> ret_pt(new rtlop_tracer());
    ret_pt -> init_ofstream(file_name);
    return ret_pt;
}

rtlop_tracer& get_tracer_pt()
{
  return *(rtlop_tracer::tracer_pt);
}

//#################################################################
extern "C" void traceIOp3(int cop_id, int rtlop_id, int op_0, int op_1, int op_out)
{
    get_tracer_pt().trace(rtlop_id, 3, op_0, op_1, op_out);
}

extern "C" void traceIOp2(int cop_id, int rtlop_id, int op_0, int op_out)
{
    get_tracer_pt().trace(rtlop_id, 2, op_0, op_out, -1);
}

extern "C" void traceIOp1(int cop_id, int rtlop_id, int op_0)
{
	get_tracer_pt().trace(rtlop_id, 1, op_0, -1, -1);
}

extern "C" void traceFOp3(int cop_id, int rtlop_id, float op_0, float op_1, float op_out)
{
    int int_op_0 = *reinterpret_cast<int*>(&op_0);
    int int_op_1 = *reinterpret_cast<int*>(&op_1);
    int int_op_2 = *reinterpret_cast<int*>(&op_out);

	get_tracer_pt().trace(rtlop_id, 3, int_op_0, int_op_1, int_op_2);
}

extern "C" void traceFIOp3(int cop_id, int rtlop_id, float op_0, float op_1, int op_out)
{
    int int_op_0 = *reinterpret_cast<int*>(&op_0);
    int int_op_1 = *reinterpret_cast<int*>(&op_1);
    
	get_tracer_pt().trace(rtlop_id, 3, int_op_0, int_op_1, op_out);
}

extern "C" void traceFOp2(int cop_id, int rtlop_id, float op_0, float op_out)
{
	int int_op_0 = *reinterpret_cast<int*>(&op_0);
    int int_op_1 = *reinterpret_cast<int*>(&op_out);

    get_tracer_pt().trace(rtlop_id, 2, int_op_0, int_op_1, -1);
}

extern "C" void traceFOp1(int cop_id, int rtlop_id, float op_0)
{
	int int_op_0 = *reinterpret_cast<int*>(&op_0);

    get_tracer_pt().trace(rtlop_id, 1, int_op_0, -1, -1);
}

//#################################################################
extern "C" float _autotb_FifoWrite_float(float* mem_pt, float wr_val)
{
	printf("%p,%.4f\n", mem_pt, wr_val);
	return wr_val;
}

extern "C" float pow_float(float pow_val)
{
    return powf(2, pow_val);
}

//#################################################################
extern "C" unsigned int _select32(int val, int lo, int hi)
{
	unsigned int v = val;
	int mask = -1;
	if(lo <= hi)
	{
		v >>= lo;
		mask >>= (31 - (hi - lo));
		return (v & mask);
	}
	else
	{
		v >>= hi;
		mask >>= (31 - (lo - hi));
		v = v & mask;
		unsigned int ret = 0;
		for(int i = 0; i < lo - hi + 1; i++)
		{
			ret |= ((v >> i) & 0x1);
			if(i < lo - hi)
			{
				ret = (ret << 1);
			}
		}
		return ret;
	}
}

extern "C" unsigned long long _select64(long long val, long long lo, long long hi)
{
	unsigned long long v = val;
	long long mask = -1;
	if(lo <= hi)
	{
		v >>= lo;
		mask >>= (63 - (hi - lo));
		return (v & mask);
	}
	else
	{
		v >>= hi;
		mask >>= (63 - (lo - hi));
		v = v & mask;
		unsigned long long ret = 0;
		for(int i = 0; i < lo - hi + 1; i++)
		{
			ret |= ((v >> i) & 0x1);
			if(i < lo - hi)
			{
				ret = (ret << 1);
			}
		}
		return ret;
	}
}

extern "C" int _set32(int val,int rep, int lo, int hi)
{
	unsigned int mask = 0xffffffff;
	if(lo <= hi)
	{
		mask >>= (31 - (hi - lo));
		mask <<= lo;
		val &= (~mask);
		rep &= (mask >> lo);
		return (val | (rep << lo));

	}
	else
	{
		mask >>= (31 - (lo - hi));
		mask <<= hi;
		val &= (~mask);
		rep &= (mask >> hi);
		int ret = 0;
		for(int i = 0; i < lo - hi + 1; i++)
		{
			ret |= ((rep >> i) & 0x1);
			if(i < lo - hi)
			{
				ret = (ret << 1);
			}
		}
		return ((ret << hi) | val);
	}
}

extern "C" long long _set64(long long val,long long rep, long long lo, long long hi)
{
	unsigned long long	mask = 0xffffffffffffffff;
	if(lo <= hi)
	{
		mask >>= (63 - (hi - lo));
		mask <<= lo;
		val &= (~mask);
		rep &= (mask >> lo);
		return (val | (rep << lo));

	}
	else
	{
		mask >>= (63 - (lo - hi));
		mask <<= hi;
		val &= (~mask);
		rep &= (mask >> hi);
		int ret = 0;
		for(int i = 0; i < lo - hi + 1; i++)
		{
			ret |= ((rep >> i) & 0x1);
			if(i < lo - hi)
			{
				ret = (ret << 1);
			}
		}
		return ((ret << hi) | val);
	}
}