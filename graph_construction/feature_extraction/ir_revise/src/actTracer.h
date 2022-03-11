#pragma once

#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "llvm/IR/Module.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/CommandLine.h"

#include "c_operator.h"

using namespace llvm;

extern int actTraceCnt;

class actTracer : public ModulePass
{
	public:
		static char ID;
		actTracer() : ModulePass(ID) {}

		std::map<string, c_operator> copMap;
		std::string kernelName;

		void init(std::string _kernelName, std::map<string, c_operator>* _copMap);
		virtual bool runOnModule(Module &M);
		virtual void getAnalysisUsage(AnalysisUsage &AU) const;

		Function* getTargetFunc(Module& M,std::string targetName);
		void createActTracer(Module &M, BasicBlock *B, Instruction *I, c_operator *cop);
		void traceI2O1(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceO1(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceI1(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceI1O1(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceFF(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceFC(BasicBlock *B, Instruction *I, c_operator *cop);
		void traceSL(BasicBlock *B, Instruction *I, c_operator *cop);
};


