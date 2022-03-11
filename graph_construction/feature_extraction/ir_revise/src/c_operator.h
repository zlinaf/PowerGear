#pragma once

#include <string>
#include <map>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <boost/algorithm/string.hpp>

using namespace std;

//###############################################################
class csv_reader
{
	string fileName;
	string delimeter;
 
public:
	csv_reader(string filename, string delm = "|") : 
	fileName(filename), delimeter(delm)
	{ }
 
	vector<vector<string> > getData();
};

//###############################################################
extern const vector<string> opcodeI2O1;
extern const vector<string> opcodeI1O1;
extern const vector<string> opcodeI1;
extern const vector<string> opcodeO1;
extern const vector<string> opcodeFF;
extern const vector<string> opcodeFC;
extern const vector<string> opcodeSL;
extern const vector<string> codeWaive;

class c_operator
{
	public:
		c_operator(int copID, vector<int> entityVec, int RTLopID, string opcode, int opndNum, int bwOut, int bw0, int bw1, int bw2, string instruction)
			: copID(copID), entityVec(entityVec), RTLopID(RTLopID), opcode(opcode), opndNum(opndNum),
			bwOut(bwOut), bw0(bw0), bw1(bw1), bw2(bw2), instruction(instruction)
		{
		}

		c_operator()
		{
			copID = -1;
			RTLopID = -1;
			opcode = "";
			opndNum = -1;
			bwOut = -1;
			bw0 = -1;
			bw1 = -1;
			bw2 = -1;
			instruction = "";
		}

		c_operator(const c_operator& cop_in)
		{
			copID = cop_in.copID;
            entityVec = cop_in.entityVec;
			RTLopID = cop_in.RTLopID;
			opcode = cop_in.opcode;
			opndNum = cop_in.opndNum;
			bwOut = cop_in.bwOut;
			bw0 = cop_in.bw0;
			bw1 = cop_in.bw1;
			bw2 = cop_in.bw2;
			instruction = cop_in.instruction;
		}

		int copID;
        vector<int> entityVec;
		int RTLopID;
		string opcode;
		int opndNum;
		int bwOut;
		int bw0;
		int bw1;
		int bw2;
		string instruction;
};

extern map<string, c_operator> copMapLocal;

string& removeSpace(string& str);
map<string, c_operator>* readCopFile(const char* copPath);
bool matchOpcode(const string &targetOpcode, const vector<string> &vecOpcode);
void trimString(string& targetStr, const string& subStr);
int bypassOpcode(const string code);
