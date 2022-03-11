
from ast import arg
import os
import csv
import re
import  os
from statistics import mean
from unittest import result
import pandas as pd
import argparse

# from ensemble.train.main import PROJECT_ROOT
PROJECT_ROOT = os.path.abspath('..')
ROOT_DIR = os.path.abspath('../../..')
print(PROJECT_ROOT)
print(ROOT_DIR)
#将所有文件的路径放入到listcsv列表中


def get_single_dataset_result(model_list,fater_path):
    num_models = len(model_list)
    result_col_name = []
    for i in range(num_models):
        result_col_name.append('y_hat_{a}'.format(a = i))
    result_col_name.append('y')
    result_df = pd.DataFrame(columns = result_col_name)
    for model_index,each_model in enumerate(model_list):
        each_model_dir = '{a}/{b}'.format(a = fater_path, b = each_model)
        with open(each_model_dir) as f:
            f_csv = pd.read_csv(f)
            f_csv = f_csv.sort_values('y')
            result_df[result_col_name[model_index]] = f_csv['y_hat'].values
            if model_index == len(model_list)-2:
                result_df['y'] = f_csv['y'].values
    return result_df



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'get ensemble model result', epilog = '')
    parser.add_argument('--paper_model', help = "get paper model result", action='store_true')
    args = parser.parse_args()
    print(PROJECT_ROOT)
    if args.paper_model:
        test_result_dir = '{a}/paper_model/test_result'.format(a = ROOT_DIR)
    else:
        test_result_dir = '{a}/test/test_result'.format(a = PROJECT_ROOT)
    dataset_list = os.listdir(test_result_dir)
    print(dataset_list)
    result_dic = {}
    final_result_list = []
    for dataset_name in dataset_list:
        single_dataset_dir = '{a}/{b}'.format(a = test_result_dir, b = dataset_name)
        single_dataset_model_result_list = os.listdir(single_dataset_dir)
        num_models = len(single_dataset_model_result_list)
        single_result = get_single_dataset_result(single_dataset_model_result_list,single_dataset_dir)
        temp_df = single_result.iloc[:,0:num_models]
        temp_df['model_mean'] = temp_df.apply(lambda x: x.sum()/num_models, axis=1)
        temp_mape = abs(temp_df['model_mean']-single_result['y'])/single_result['y']
        temp_mape = temp_mape[temp_mape>0]
        result_dic[dataset_name] = mean(temp_mape)
        final_result_list.append(mean(temp_mape))
    flog = open('{}/result.log'.format(os.path.abspath('')), 'a+', newline = '')
    print("#######result_dic#######")
    flog.write("#######result_dic#######\n")
    print(result_dic)
    flog.write(str(result_dic)+'\n')
    print("#######final mape#######")
    flog.write("#######final mape#######"+'\n')
    print(mean(final_result_list))
    flog.write(str(mean(final_result_list))+'\n')
    flog.close()


    