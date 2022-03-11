# Ensemble Model
The code to develop an ensemble HEC-GNN model.

## Parameters and Defaut Values

	--layer_num 3
	--hidden_dim 128  # hidden layer dimension
	--batch_size 128
    --learning_rate	0.0005
    --use_overall 1   # whether to use overall attr: 1 or 0
    --drop_out 0.2
    --relations 4     # how many relation types in graph. options:[1,4,8]
    --edge_dim 4       # edge attribute dimension
    --overall_dim_large 128      # hidden layer dimension of overall attribute.
	--edge_feature 1   # use edge attribute. options:[0,1]
    --node_feature 1   # use node attribute. options:[0,1]
    --test_dataset atax   # which dataset is test dataset. options:["atax","bicg","gemm","gesummv","k2mm","k3mm","mvt","syr2k","syrk"]
    --train_dataset all  # which training mode
    --seed    # seed number
	--onevone 0
	--aggr_type add    # aggregate function type. options:["add","mean"]
    --pool_type add  # pooling layer tyeo. options:["add","mean"]
    --k 10            # how many folds to split val set and train set.   
    --fold_index 9    # the index of val fold. fold_index <= k-1
	--loss_function mape
	--JK sum  # Jumping Knowledge. options:["last","sum","max","lstm"]

### Paper Model
Our model is stored in the folder `paper_model`. Among them, `trained_model` is the model corresponding to each dataset, and `test_result` is the test result corresponding to each model. We can directly obtain the results in the paper through the code in `get_ensemble_result`.

### Get Paper Model Result
Running the following code can get the results of the ensemble model of the model trained in our paper.

	cd get_ensemble_result; python get_result.py --paper_model;

### Train
	cd train; sh train.sh;

### Test
	cd test; CUDA_VISIBLE_DEVICES=0 python main.py --k 10 --fold_index 10 --seed_number_list 1 2 3 
The parameters should same as `train.sh`. `fold_index` should equal to `k`.

### Get Emsemble Result
If you get your own trained model and get test results, run the following code to get the final ensemble model results.

	cd get_ensemble_result; python get_result.py;

This step must run after training and testing.


	
