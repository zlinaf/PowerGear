#include "actTracer.h"

char actTracer::ID = 0; // LLVM uses ID’s address to identify a pass, so initialization value is not important.
static RegisterPass<actTracer> A("actTracer", "activity trace function");

int actTraceCnt = 0;

//###############################################################
void actTracer::init(std::string _kernelName, std::map<string, c_operator> *_copMap)
{
	kernelName = _kernelName;
	copMap = *_copMap;
}

//###############################################################
bool actTracer::runOnModule(Module &M)
{
	if (copMap.size() == 0)
		assert(false && "actTracer::runOnModule: copMap.size() == 0");
	
	Function *F = getTargetFunc(M, kernelName);

	std::set<string> instTraced;

	for (auto &B : *F)
	{
		for (auto &I : B)
		{
			c_operator *cop;
			std::string strTmp;
			raw_string_ostream ostr(strTmp);
			ostr << I;
			std::string IStr = removeSpace(ostr.str());

			if (bypassOpcode(IStr) == 0 && copMap.find(IStr) != copMap.end())
			{
				if (instTraced.find(IStr) == instTraced.end())
					instTraced.insert(IStr);

				actTraceCnt++;
				cop = &copMap[IStr];
				createActTracer(M, &B, &I, cop);
			}
		}
	}

	// for check
	for (auto it : copMap)
	{
		if (instTraced.find(it.first) == instTraced.end())
		{
			errs() << "CHECK: trace failure: " << "keycode = " << it.first << ", instruction = " << it.second.instruction << "\n";
		}
	}
	
	errs() << "\n\n### IR code for annotated target function ###" << "\n";
	for (auto &B : *F)
		for (auto &I : B)
			errs() << I << "\n";

	return true;
}

//###############################################################
void actTracer::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
	AU.addRequiredTransitive<LoopInfoWrapperPass>();
}

//###############################################################
Function* actTracer::getTargetFunc(Module &M, std::string targetName)
{
	Function* target = M.getFunction(StringRef(targetName.c_str()));
	if(target == nullptr)
	{
		errs() << "Cannot find the target function = " << targetName << "\n";
		assert(false && "target function should be found");
	}
	return target;
}

//###############################################################
void actTracer::createActTracer(Module &M, BasicBlock *B, Instruction *I, c_operator *cop)
{

	std::vector<std::vector<Value*>> args(cop->entityVec.size());

	if (matchOpcode(cop->opcode, opcodeI2O1)) // {"add", "sub", "mul", "div", "fadd", "fsub", "fmul", "fdiv", "icmp", "and", "or", "xor"}
	{
		traceI2O1(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeI1O1)) // {"sqrt", "fsqrt"}
	{
		traceI1O1(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeO1)) // {"mux", "load", "read"}
	{
		traceO1(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeI1)) // {"store"}
	{
		traceI1(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeFF)) // {"write"}
	{
		traceFF(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeFC)) // {"fcmp"}
	{
		traceFC(B, I, cop);
	}
	else if (matchOpcode(cop->opcode, opcodeSL)) // {"select"}
	{
		traceSL(B, I, cop);
	}
	else{
		 errs() << "CHECK: no trace function matching for copID = " << cop->copID << ", opcode = " << cop->opcode << "\n";
	}
}

//###############################################################
void actTracer::traceI2O1(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt0;
	Value *newSExt1;
	Value *newSExtOut;

	//##############################
	if (I->getOperand(0)->getType()->isIntegerTy() && cop->bw0 != 32)
	{
		newSExt0 = Builder.CreateZExtOrTrunc(I->getOperand(0), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt0))
        {
            Instruction *tracer0 = cast<Instruction>(newSExt0);
            Builder.SetInsertPoint(tracer0->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt0 = I->getOperand(0);
	}

	if (I->getOperand(1)->getType()->isIntegerTy() && cop->bw1 != 32)
	{
		newSExt1 = Builder.CreateZExtOrTrunc(I->getOperand(1), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt1))
        {
            Instruction *tracer1 = cast<Instruction>(newSExt1);
            Builder.SetInsertPoint(tracer1->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt1 = I->getOperand(1);
	}

	if (I->getType()->isIntegerTy() && cop->bwOut != 32)
	{
		newSExtOut = Builder.CreateZExtOrTrunc(I, llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExtOut))
        {
            Instruction *tracerOut = cast<Instruction>(newSExtOut);
            Builder.SetInsertPoint(tracerOut->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExtOut = I;
	}

	//##############################
	std::string tracerName;
	if (newSExtOut->getType()->isIntegerTy())
		tracerName = "traceIOp3";
	else if (newSExtOut->getType()->isFloatTy())
		tracerName = "traceFOp3";
	else
		assert(false && "the IN2OUT1 tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt0->getType());
		ArgTys.push_back(newSExt1->getType());
		ArgTys.push_back(newSExtOut->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt0);
		Args.push_back(newSExt1);
		Args.push_back(newSExtOut);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the IN2OUT1 tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);
		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}

//###############################################################
void actTracer::traceI1O1(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt0;
	Value *newSExtOut;

	//##############################
	if (I->getOperand(0)->getType()->isIntegerTy() && cop->bw0 != 32)  // TODO: need to reconsider the case with the bitwidth = 64 (long type)
	{
		newSExt0 = Builder.CreateZExtOrTrunc(I->getOperand(0), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt0))
        {
            Instruction *tracer0 = cast<Instruction>(newSExt0);
            Builder.SetInsertPoint(tracer0->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt0 = I->getOperand(0);
	}

	if (I->getType()->isIntegerTy() && cop->bwOut != 32)
	{
		newSExtOut = Builder.CreateZExtOrTrunc(I, llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExtOut))
        {
            Instruction *tracerOut = cast<Instruction>(newSExtOut);
            Builder.SetInsertPoint(tracerOut->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExtOut = I;
	}

	//##############################
	std::string tracerName;
	if (newSExtOut->getType()->isIntegerTy())
    	tracerName = "traceIOp2";
	else if (newSExtOut->getType()->isFloatTy())
		tracerName = "traceFOp2";
	else
		assert(false && "the IN1OUT1 tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 4> ArgTys;
		SmallVector<Value *, 4> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt0->getType());
		ArgTys.push_back(newSExtOut->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt0);
		Args.push_back(newSExtOut);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the IN1OUT1 tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);
		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
	
}

//###############################################################
void actTracer::traceO1(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExtOut;

	//##############################
	if (I->getType()->isIntegerTy() && cop->bwOut != 32)
	{
		newSExtOut = Builder.CreateZExtOrTrunc(I, llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExtOut))
        {
            Instruction *tracerOut = cast<Instruction>(newSExtOut);
            Builder.SetInsertPoint(tracerOut->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExtOut = I;
	}

	//##############################
	std::string tracerName;
	if (newSExtOut->getType()->isIntegerTy())
		tracerName = "traceIOp1";
	else if (newSExtOut->getType()->isFloatTy())
		tracerName = "traceFOp1";
	else
		assert(false && "the OUT1 tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExtOut->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExtOut);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the OUT1 tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);
		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}

//###############################################################
void actTracer::traceI1(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt0;

	//##############################
	if (I->getOperand(0)->getType()->isIntegerTy() && cop->bw0 != 32)  // TODO: need to reconsider the case with the bitwidth = 64 (long type)
	{
		newSExt0 = Builder.CreateZExtOrTrunc(I->getOperand(0), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt0))
        {
            Instruction *tracer0 = cast<Instruction>(newSExt0);
            Builder.SetInsertPoint(tracer0->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt0 = I->getOperand(0);
	}

	//##############################
	std::string tracerName; // TODO: try whether it can do function overload, so no need to separately consider float/integer types
	if (newSExt0->getType()->isIntegerTy())
    	tracerName = "traceIOp1";
	else if (newSExt0->getType()->isFloatTy())
		tracerName = "traceFOp1";
	else
		assert(false && "the IN1 tracer function are neither integer type or floating type.");
	
	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt0->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt0);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the IN1 tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);
		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}

//###############################################################
void actTracer::traceFF(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt1;

	//##############################
	if (I->getOperand(1)->getType()->isIntegerTy() && cop->bw1 != 32)  // TODO: need to reconsider the case with the bitwidth = 64 (long type)
	{
		newSExt1 = Builder.CreateZExtOrTrunc(I->getOperand(1), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt1))
        {
            Instruction *tracer1 = cast<Instruction>(newSExt1);
            Builder.SetInsertPoint(tracer1->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt1 = I->getOperand(1);
	}

	//##############################
	std::string tracerName;
	if (newSExt1->getType()->isIntegerTy())
    	tracerName = "traceIOp1";
	else if (newSExt1->getType()->isFloatTy())
		tracerName = "traceFOp1";
	else
		assert(false && "the FF tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt1->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt1);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the FF tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);
		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}

//###############################################################
void actTracer::traceFC(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt0;
	Value *newSExt1;
	Value *newSExtOut;

	//##############################
	// the two input operands are floating point, output is integer
	newSExt0 = I->getOperand(0);  
	newSExt1 = I->getOperand(1);

	if (I->getType()->isIntegerTy() && cop->bwOut != 32)
	{
		newSExtOut = Builder.CreateZExtOrTrunc(I, llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExtOut))
        {
            Instruction *tracerOut = cast<Instruction>(newSExtOut);
            Builder.SetInsertPoint(tracerOut->getNextNode());
        }
	}
	else // operand integer bitwidth = 32
	{
		newSExtOut = I;
	}

	//##############################
	std::string tracerName;
	if (newSExt0->getType()->isFloatTy() && newSExt1->getType()->isFloatTy() && newSExtOut->getType()->isIntegerTy())
		tracerName = "traceFIOp3";
	else
		assert(false && "the FCMP tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt0->getType());
		ArgTys.push_back(newSExt1->getType());
		ArgTys.push_back(newSExtOut->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt0);
		Args.push_back(newSExt1);
		Args.push_back(newSExtOut);
		
		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the FCMP tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);

		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}

//###############################################################
void actTracer::traceSL(BasicBlock *B, Instruction *I, c_operator *cop)
{
	IRBuilder<> Builder(I->getNextNode());
	Value *newSExt0;
	Value *newSExt1;
	Value *newSExtOut;

	//##############################
	if (I->getOperand(1)->getType()->isIntegerTy() && cop->bw1 != 32)  // TODO: need to reconsider the case with the bitwidth = 64 (long type)
	{
		newSExt0 = Builder.CreateZExtOrTrunc(I->getOperand(1), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt0))
        {
            Instruction *tracer0 = cast<Instruction>(newSExt0);
            Builder.SetInsertPoint(tracer0->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt0 = I->getOperand(1);
	}

	if (I->getOperand(2)->getType()->isIntegerTy() && cop->bw2 != 32)
	{
		newSExt1 = Builder.CreateZExtOrTrunc(I->getOperand(2), llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExt1))
        {
            Instruction *tracer1 = cast<Instruction>(newSExt1);
            Builder.SetInsertPoint(tracer1->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExt1 = I->getOperand(2);
	}

	if (I->getType()->isIntegerTy() && cop->bwOut != 32)
	{
		newSExtOut = Builder.CreateZExtOrTrunc(I, llvm::Type::getInt32Ty(B->getContext()));
		if (isa<Instruction>(newSExtOut))
        {
            Instruction *tracerOut = cast<Instruction>(newSExtOut);
            Builder.SetInsertPoint(tracerOut->getNextNode());
        }
	}
	else // operand integer bitwidth = 32 || operand type = float
	{
		newSExtOut = I;
	}

	//##############################
	std::string tracerName; // TODO: try whether it can do function overload, so no need to separately consider float/integer types
	if (newSExtOut->getType()->isIntegerTy())
    	tracerName = "traceIOp3";
	else if (newSExtOut->getType()->isFloatTy())
		tracerName = "traceFOp3";
	else
		assert(false && "the IN2OUT1 tracer function are neither integer type or floating type.");

	for(int i = 0; i < cop->entityVec.size(); i++)
	{
		int copID = cop->copID;
		int entityID = cop->entityVec[i];

		SmallVector<llvm::Type *, 5> ArgTys;
		SmallVector<Value *, 5> Args;
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(B->getContext()));
		ArgTys.push_back(newSExt0->getType());
		ArgTys.push_back(newSExt1->getType());
		ArgTys.push_back(newSExtOut->getType());
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), copID));
		Args.push_back(ConstantInt::get(llvm::Type::getInt32Ty(B->getContext()), entityID));
		Args.push_back(newSExt0);
		Args.push_back(newSExt1);
		Args.push_back(newSExtOut);

		Value *newFuncVal = B->getParent()->getParent()->getOrInsertFunction(tracerName, FunctionType::get(llvm::Type::getVoidTy(B->getContext()), ArgTys, false)).getCallee();
		Function *newFunc = dyn_cast<Function>(newFuncVal);
		assert(newFunc && "the IN2OUT1 tracer function can not been created successfully.");
		Value *newTracer = Builder.CreateCall(newFunc->getFunctionType(), newFuncVal, Args);

		Instruction *tracerPt = cast<Instruction>(newTracer);
        Builder.SetInsertPoint(tracerPt->getNextNode());
	}
}
