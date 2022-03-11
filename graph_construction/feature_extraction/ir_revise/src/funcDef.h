#pragma once

#include <stdio.h>
#include <stdlib.h>
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

using namespace llvm;

class funcDef : public ModulePass {
	public:
		static char ID;
		funcDef() : ModulePass(ID) {}

		virtual void getAnalysisUsage(AnalysisUsage &AU) const;
		virtual bool runOnModule(Module &M);
		std::string setFuncName(std::string name);
};

