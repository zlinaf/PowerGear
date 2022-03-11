#include "rtlop_tracer.h"

//#################################################################
float compute_hamm_dist(int curr, int prev)
{
    bitset<BW> bin_xor(curr ^ prev);
    return bin_xor.count();
}

float average_hamm_dist(int cnt, float avg, float dist)
{
    float new_avg;
    new_avg = ((cnt - 2) * avg + dist) / float(cnt - 1);
    return new_avg;
}

//#################################################################
unique_ptr<rtlop_tracer> rtlop_tracer::tracer_pt;

rtl_operator::rtl_operator(int op_id) : rtlop_id(op_id)
{
    count = 0;
}

bool rtl_operator::init(int opnd_num, int op_0, int op_1, int op_2)
{
    if((opnd_num < 1) || (opnd_num > 3))
    {
        cout << "CHECK in rtl_operator::init: operand number (" << opnd_num << ") not in range [1, 3] for rtlop_id (" << rtlop_id << ") \n";
        return false;
    }
    else
    {
        count = 1;
        hamm_dist.clear();
        prev_val.clear();
        int op[3] = {op_0, op_1, op_2};
        for(int i = 0; i < opnd_num; i++)
        {
            hamm_dist.push_back(0);
            prev_val.push_back(op[i]);
        }
        return true;
    }
}

bool rtl_operator::update(int opnd_num, int op_0, int op_1, int op_2)
{
    if((opnd_num != hamm_dist.size()) || (opnd_num < 1) || (opnd_num > 3))
    {
        cout << "CHECK in rtl_operator::update: operand number (" << opnd_num << ") not in range [1, 3] / not matching op size for rtlop_id (" << rtlop_id << ") \n";
        return false;
    }
    else
    {
        count++;
        int op[3] = {op_0, op_1, op_2};
        float dist;
        for(int i = 0 ; i < opnd_num; i++)
        {
            dist = compute_hamm_dist(op[i], prev_val[i]);
            hamm_dist[i] = average_hamm_dist(count, hamm_dist[i], dist);
            prev_val[i] = op[i];
        }
        return true;
    }
}

//#################################################################
rtlop_tracer::~rtlop_tracer()
{
    if(fout != nullptr)
        fout->close();
    if(rtlop_map.size() > 0)
        rtlop_map.clear();
}

void rtlop_tracer::init_ofstream(const string file_name)
{
    if(file_name != "")
	{
		fout = new ofstream(file_name, ios_base::out);
	}
    else
    {
        fout = nullptr;
    }
}

void rtlop_tracer::trace(int rtlop_id, int opnd_num, int op_0, int op_1, int op_2){
    map<int, rtl_operator>::iterator iter;
    iter = rtlop_map.find(rtlop_id);
    if(iter != rtlop_map.end())
    {
        rtl_operator& rtlop = (*iter).second;
        rtlop.update(opnd_num, op_0, op_1, op_2);
    }
    else
    {
        rtl_operator rtlop(rtlop_id);
        rtlop.init(opnd_num, op_0, op_1, op_2);
        rtlop_map.insert(pair<int, rtl_operator>(rtlop_id, rtlop));
    }
}

void rtlop_tracer::print()
{
    *fout << "rtlop_id,count,hamm_dist[0],hamm_dist[1],hamm_dist[2]" << "\n";
    rtl_operator* rtlop;
    map<int, rtl_operator>::iterator op_iter;

    for(op_iter = rtlop_map.begin(); op_iter != rtlop_map.end(); op_iter++)
    {
        rtlop = &(op_iter->second);
        *fout << rtlop->rtlop_id << "," << rtlop->count << ",";

        for(int i = 0; i < rtlop->hamm_dist.size(); i++)
        {
            if(i == rtlop->hamm_dist.size()-1)
            {
                *fout << rtlop->hamm_dist[i] << "\n";
            }
            else
            {
                *fout << rtlop->hamm_dist[i] << ",";
            }
        }
    }
}
