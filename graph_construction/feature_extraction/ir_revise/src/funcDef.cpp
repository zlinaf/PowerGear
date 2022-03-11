#include "funcDef.h"

char funcDef::ID = 0; // LLVM uses ID’s address to identify a pass, so initialization value is not important.
static RegisterPass<funcDef> P("funcDef", "external function declare and define");

//###############################################################
bool funcDef::runOnModule(Module &M) {
	std::vector<std::string> llvmSelectName;
	std::vector<std::string> llvmSetName;
	std::vector<Function *> llvmSelectFunc;
	std::vector<Function *> llvmSetFunc;
	std::vector<Function *> externSelectFunc;
	std::vector<Function *> externSetFunc;

	Function *funcSelect32;
	Function *funcSet32;
	Function *funcSelect64;
	Function *funcSet64;

	//###################################
	for (auto &F : M)
	{
		if (F.getName().find("llvm.part.select") != std::string::npos)
			llvmSelectName.push_back(F.getName().str());

		if (F.getName().find("llvm.part.set") != std::string::npos)
			llvmSetName.push_back(F.getName().str());

		if (F.getName().find("pow_generic") != std::string::npos)
		{
			F.setName("pow_float");
			F.deleteBody();
		}
	}

	//################# Declare functions ##################
	Value *newFuncVal;

	if (llvmSelectName.size() > 0)
	{
		SmallVector<llvm::Type *, 3> ArgTys;
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		newFuncVal = M.getOrInsertFunction("_select32", FunctionType::get(llvm::Type::getInt32Ty(M.getContext()), ArgTys, false)).getCallee();
		funcSelect32 = dyn_cast<Function>(newFuncVal);
		
		ArgTys.clear();
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		newFuncVal = M.getOrInsertFunction("_select64", FunctionType::get(llvm::Type::getInt64Ty(M.getContext()), ArgTys, false)).getCallee();
		funcSelect64 = dyn_cast<Function>(newFuncVal);
	}

	if(llvmSetName.size() > 0)
	{
		SmallVector<llvm::Type *, 4> ArgTys;
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt32Ty(M.getContext()));
		newFuncVal = M.getOrInsertFunction("_set32", FunctionType::get(llvm::Type::getInt32Ty(M.getContext()), ArgTys, false)).getCallee();
		funcSet32 = dyn_cast<Function>(newFuncVal);
		
		ArgTys.clear();
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		ArgTys.push_back(llvm::Type::getInt64Ty(M.getContext()));
		newFuncVal = M.getOrInsertFunction("_set64", FunctionType::get(llvm::Type::getInt64Ty(M.getContext()), ArgTys, false)).getCallee();
		funcSet64 = dyn_cast<Function>(newFuncVal);
	}

	//#################### Define wrapper functions ###############
	for(int i = 0; i < llvmSelectName.size(); i++)
	{
		Function *sel_func = M.getFunction(llvmSelectName[i]);
		llvmSelectFunc.push_back(sel_func);
		FunctionType* sel_ty = sel_func->getFunctionType();
		std::string newFuncName = setFuncName(llvmSelectName[i]);
		Value *newFuncVal = M.getOrInsertFunction(newFuncName, sel_ty).getCallee();
    	Function *newFunc = dyn_cast<Function>(newFuncVal);
		externSelectFunc.push_back(newFunc);
		Function::arg_iterator args = newFunc->arg_begin();
		Value* v = args++;
		v->setName("v");
		Value* lo = args++;
		lo->setName("lo");
		Value* hi = args++;
		hi->setName("hi");
		BasicBlock* block = BasicBlock::Create(M.getContext(), "entry", newFunc);
		IRBuilder<> builder(block);
		if(sel_func->getReturnType()->getIntegerBitWidth() <= 32)
		{
			Type *int32Ty = Type::getInt32Ty(M.getContext());
			Value *selArgs[] = {
				builder.CreateZExtOrTrunc(v, int32Ty),
				builder.CreateZExtOrTrunc(lo, int32Ty),
				builder.CreateZExtOrTrunc(hi, int32Ty)
			};
			Value *call = builder.CreateCall(funcSelect32, ArrayRef<Value*>(selArgs));
			Value *ret = builder.CreateZExtOrTrunc(call, v->getType());
			builder.CreateRet(ret);
		}
		else
		{
			Type* int64Ty = Type::getInt64Ty(M.getContext());
			Value* selArgs[] = {
				builder.CreateZExtOrTrunc(v,int64Ty),
				builder.CreateZExtOrTrunc(lo,int64Ty),
				builder.CreateZExtOrTrunc(hi,int64Ty)
			};
			Value* call = builder.CreateCall(funcSelect64, ArrayRef<Value*>(selArgs));
			Value* ret = builder.CreateZExtOrTrunc(call, v->getType());
			builder.CreateRet(ret);
		}
	}

	for(int i = 0; i < llvmSetName.size(); i++)
	{
		Function* set_func = M.getFunction(llvmSetName[i]);
		llvmSetFunc.push_back(set_func);
		FunctionType* sel_ty = set_func->getFunctionType();
		std::string newFuncName = setFuncName(llvmSetName[i]);
		Value *newFuncVal = M.getOrInsertFunction(newFuncName, sel_ty).getCallee();
    	Function *newFunc = dyn_cast<Function>(newFuncVal);

		externSetFunc.push_back(newFunc);
		Function::arg_iterator args = newFunc->arg_begin();
		Value* v = args++;
		v->setName("v");
		Value* rep = args++;
		rep->setName("rep");
		Value* lo = args++;
		lo->setName("lo");
		Value* hi = args++;
		hi->setName("hi");
		BasicBlock* block = BasicBlock::Create(M.getContext(), "entry", newFunc);
		IRBuilder<> builder(block);
		if(set_func->getReturnType()->getIntegerBitWidth() <= 32)
		{
			Type* int32Ty = Type::getInt32Ty(M.getContext());
			Value* selArgs[] = {
				builder.CreateZExtOrTrunc(v,int32Ty),
				builder.CreateZExtOrTrunc(rep,int32Ty),
				builder.CreateZExtOrTrunc(lo,int32Ty),
				builder.CreateZExtOrTrunc(hi,int32Ty)
			};
			Value* call = builder.CreateCall(funcSet32, ArrayRef<Value*>(selArgs));
			Value* ret = builder.CreateZExtOrTrunc(call, v->getType());
			builder.CreateRet(ret);
		}
		else
		{
			Type* int64Ty =Type::getInt64Ty(M.getContext());
			Value* selArgs[] = {
				builder.CreateZExtOrTrunc(v,int64Ty),
				builder.CreateZExtOrTrunc(rep,int64Ty),
				builder.CreateZExtOrTrunc(lo,int64Ty),
				builder.CreateZExtOrTrunc(hi,int64Ty)};
			Value* call = builder.CreateCall(funcSet64, ArrayRef<Value*>(selArgs));
			Value* ret = builder.CreateZExtOrTrunc(call, v->getType());
			builder.CreateRet(ret);
		}
	}

	//###################################
	for(Module::iterator i = M.begin(), em = M.end(); i != em; i++)
		for(Function::iterator j = i->begin(), ef = i->end(); j != ef; j++)
			for(BasicBlock::iterator k = j->begin(), eb = j->end(); k != eb; k++)
			{
				if(CallInst* call_inst = dyn_cast<CallInst>(&*k))
				{
					for(int l = 0; l < llvmSelectFunc.size(); l++)
					{
						if(call_inst->getCalledFunction() == llvmSelectFunc[l])
						{
							call_inst->setCalledFunction(externSelectFunc[l]);
						}
					}
					for(int l = 0; l < llvmSetFunc.size(); l++)
					{
						if(call_inst->getCalledFunction() == llvmSetFunc[l])
						{
							call_inst->setCalledFunction(externSetFunc[l]);
						}
					}
				}
			}

	return false;
}

//###############################################################
void funcDef::getAnalysisUsage(AnalysisUsage &AU) const {
	AU.setPreservesAll();
}

//###############################################################
std::string funcDef::setFuncName(std::string name)
{
	std::string newName = name.substr(10);

	for (int i = 0; i < newName.size(); i++)
	{
		if (newName[i] == '.')
			newName[i] = '_';
	}

	return newName;
}