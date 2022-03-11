#include "c_operator.h"

//###############################################################
const vector<string> opcodeI2O1 = {"add", "sub", "mul", "div", "fadd", "fsub", "fmul", "fdiv", "icmp", "and", "or", "xor"};
const vector<string> opcodeI1O1 = {"sqrt", "fsqrt"};
const vector<string> opcodeI1 = {"store"};
const vector<string> opcodeO1 = {"mux", "load", "read"};
const vector<string> opcodeFF = {"write"};
const vector<string> opcodeFC = {"fcmp"};
const vector<string> opcodeSL = {"select"};

map<string, c_operator> copMapLocal;

//###############################################################
string& removeSpace(string& str)
{
	string::iterator endPos = remove(str.begin(), str.end(), ' ');
	str.erase(endPos, str.end());
	endPos = find(str.begin(), str.end(), '#');
	str.erase(endPos, str.end());
	return str;
}

//###############################################################
vector<vector<string>> csv_reader::getData()
{
	ifstream ifile(fileName);
	vector<vector<string>> dataList;
	string line = "";
	int lineCnt = 0;

	while (getline(ifile, line)){
		vector<string> vec;
		boost::algorithm::split(vec, line, boost::is_any_of(delimeter));
		dataList.push_back(vec);
	}
	ifile.close();
	return dataList;
}

//###############################################################
vector<int> getEntityVec(string str){
    vector<int> entityVec;
    vector<string> entityStrVec;
	boost::algorithm::split(entityStrVec, str, boost::is_any_of(","));
    for(string edge_id : entityStrVec){
        entityVec.push_back(stoi(edge_id));
    }
    return entityVec;
}

//###############################################################
map<string, c_operator>* readCopFile(const char* copPath){
    int copID, RTLopID, opndNum, bwOut, bw0, bw1, bw2;
    string opcode, keyCode, instruction;
    vector<int> entityVec;

    csv_reader csvrd(copPath);
    vector<vector<string>> copFile = csvrd.getData();

    for (vector<string> copLine : copFile){
		copID = stoi(copLine[0]);
        entityVec = getEntityVec(copLine[2]);
		RTLopID = stoi(copLine[3]);
		opcode = copLine[6];
		opndNum = stoi(copLine[9]);
		bwOut = stoi(copLine[10]);
		bw0 = stoi(copLine[11]);
		bw1 = stoi(copLine[12]);
		bw2 = stoi(copLine[13]);
		trimString(copLine[15], "nounwind");
		instruction = copLine[15];

		// LLVM11 has changed the LOAD instruction v.s. the version in VivadoHLS, so here need to change the instruction code accordingly
		if (copLine[15].find("= load ") != -1)
		{
			int startIndex = copLine[15].find("= load ") + 7;
			int endIndex = copLine[15].find("*");
			string opType = string(&copLine[15][startIndex], &copLine[15][endIndex]);
			instruction = copLine[15].insert(startIndex, opType + ", ");

			// cout << "after load check: " << instruction << "\n";
		}

		string instructionToTrim = instruction;
		keyCode = removeSpace(instructionToTrim);
		copMapLocal[keyCode] = c_operator(copID, entityVec, RTLopID, opcode, opndNum, bwOut, bw0, bw1, bw2, instruction);
	}

	map<string, c_operator>* copMapPtr;
	copMapPtr = &copMapLocal;
	return copMapPtr;
}

//###############################################################
bool matchOpcode(const string &targetOpcode, const vector<string> &vecOpcode)
{
	return find(vecOpcode.begin(), vecOpcode.end(), targetOpcode) != vecOpcode.end();
}

//###############################################################
const vector<string> codeWaive = {"void@trace", "brlabel", "=getelementptr", "=zext", "=bitcast", "=trunc", "@_ssdm_op_BitConcatenate",
  "=alloca", "=phi", "=sext", "@_ssdm_op_SpecLoopTripCount", "@_ssdm_op_SpecBitsMap"};

int bypassOpcode(const string code)
{
	for (auto &iter: codeWaive)
	{
		size_t findPos = code.find(iter);
		if (findPos != string::npos){
			return 1;
		}
	}
	return 0;
}

//###############################################################
void trimString(string& targetStr, const string& subStr)
{
	size_t pos = targetStr.find(subStr);
 
	if (pos != std::string::npos)
		targetStr.erase(pos, subStr.length());
}