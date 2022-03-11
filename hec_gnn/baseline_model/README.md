# Baseline Models for Comparison
The baseline GNN models for comparison in the experiments.

## Parameters and Default Values

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
    --k 5            # how many folds to split val set and train set.   
    --fold_index 4    # the index of val fold. fold_index <= k-1
	--loss_function mape
	--JK sum  # Jumping Knowledge. options:["last","sum","max","lstm"]
    --gnn_type GCN #select baseline model. options:["GCN","GIN","GINE","GraphSage","GraphConv"]

## Train and Test
	python main.py --gnn_type GCN
