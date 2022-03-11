/* #####################################
	argv[1]: kernelName
	argv[2]: input .bc file
	argv[3]: output .bc file
	argv[4]: copMap.csv
#####################################*/

#include <ios>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/SourceMgr.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <array>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <sys/time.h>

#include "llvm/Support/SourceMgr.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Pass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Support/CommandLine.h"

#include "c_operator.h"
#include "actTracer.h"
#include "funcDef.h"

using namespace llvm;

# define CLOCKS_PER_MS  ((std::clock_t) 1000)

// ####################################################
int main(int argc,char* argv[])
{
	if(argc < 4)
	{
		errs() << "Arguements are not complete.\n";
		exit(1);
	}

	errs() << "argv[1]: " << argv[1] <<"\n";
	errs() << "argv[2]: " << argv[2] <<"\n";
	errs() << "argv[3]: " << argv[3] <<"\n";
	errs() << "argv[4]: " << argv[4] <<"\n";

	std::clock_t programStart = std::clock();

	std::string kernelName = argv[1];

	SMDiagnostic Err;
	LLVMContext Context;
	std::unique_ptr<Module> Mod(parseIRFile(argv[2], Err, Context));

	std::error_code EC;
	raw_fd_ostream fout(argv[3], EC, llvm::sys::fs::F_None);

	std::map<string, c_operator>* copMap = readCopFile(argv[4]);
	int copMapCnt = copMap->size();
	errs() << "copMap->size(): " << copMap->size() << "\n";

	if (Mod)
	{
		legacy::PassManager PM;

		PM.add(new LoopInfoWrapperPass());

		auto actTrace = new actTracer();
		actTrace->init(kernelName, copMap);
		PM.add(actTrace);

		PM.add(new funcDef());

		PM.run(*Mod);
		WriteBitcodeToFile(*Mod, fout);
		
		std::cout << "Activity tracing is SUCCESSFUL.\n";

		if(copMapCnt != actTraceCnt)
			std::cout << "CHECK: copMapCnt = " << copMapCnt << ", not equal to actTraceCnt = " << actTraceCnt << "\n";
		else
			std::cout << "copMapCnt = actTraceCnt = " << actTraceCnt << ".\n";
	}
	else
	{
		std::cout << "Null module" << "\n";
	}

	std::clock_t programEnd = std::clock();
	long double programTimeMs = (programEnd - programStart) / CLOCKS_PER_MS;
	std::cout << "Program runtime = " << programTimeMs << " ms\n";
}

