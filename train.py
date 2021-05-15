# -*- coding: utf-8 -*-

from FINDER import FINDER
import sys
import os
from shutil import copy
import datetime



def save_best_model(dqn):
    print("Searching for best model...")
    vcfile_path = None
    best_model, vcfile_path, min_tour_length = dqn.findModel(VCFile_path=vcfile_path)
    min_tour_length = ''.join(min_tour_length.split('.'))
    
    print("Saving best model...")
    base_file_name = best_model.split('/')[-1]
    g_type = best_model.split('/')[-2]
    # base path to the model files and the vcmodel file
    base_path = '/'.join(vcfile_path.split('/')[0:-1]) + '/'

    files = os.listdir(base_path)
    best_model_files = []
    for f in files:
        if base_file_name in f:
            best_model_files.append(f)
    node_range = '_'.join(base_file_name.split('_')[0:3])
    target = './best_models/{}/{}/'.format(g_type, node_range + '_len_' + min_tour_length)
    try:
        os.makedirs(target)
    except:
        pass
    for file_name in best_model_files:
        new_file_name = file_name.split('.')
        new_file_name[-3] = new_file_name[-3] + '_len_' + min_tour_length
        new_file_name = '.'.join(new_file_name)
        copy(base_path + file_name, target + new_file_name)
    
    print("Saving VCFile...")
    # modify only ne name not the file ending..
    vcfile_name = vcfile_path.split('/')[-1].split('.') 
    vcfile_name[-2] = vcfile_name[-2] + '_len_' + min_tour_length
    vcfile_name = '.'.join(vcfile_name)
    copy(vcfile_path, target + vcfile_name)

    print("Saving hyperparameters and architecture...")
    code_file = open('FINDER.pyx', 'r')
    code_lines = code_file.readlines()
    code_file.close()
    relevant_lines = []
    start = False
    for k, line in enumerate(code_lines):
        if '#################################### Hyper Parameters start ####################################' in line:
            start = True
        elif '#################################### Hyper Parameters end ####################################' in line:
            start = False
        elif '###################################################### BuildNet start ######################################################' in line:
            start = True
        elif '###################################################### BuildNet end ######################################################' in line:
            start = False
        if start == True:
            relevant_lines.append(line)
    with open(target + vcfile_name.split('.')[-2] + '_architecture.pyx', 'w') as f:
        f.writelines((relevant_lines))    

    print('Saving FINDER.pyx, PrepareBatchGraph.pyx, PrepareBatchGraph.pxd, PrepareBatchGraph.h, PrepareBatchGraph.cpp...')
    file_paths = ['FINDER.pyx', 'PrepareBatchGraph.pyx', 'PrepareBatchGraph.pxd', 'src/lib/PrepareBatchGraph.cpp', 'src/lib/PrepareBatchGraph.h']
    for file_path in file_paths:
        copy(file_path, target + file_path.split('/')[-1])


def main():
    print("Starting FINDER...")
    dqn = FINDER()
    dqn.Train()
    if str(sys.argv[1]) == 'save':
        save_best_model(dqn)

if __name__=="__main__":
    main()

