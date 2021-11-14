import numpy as np
import os
from shutil import copy

def find_best_model(dqn, VCFile_path):
    if VCFile_path:
        VCFile = VCFile_path
    else:
        VCFile = './logs/%s/ModelVC_%d_%d.csv'%(dqn.cfg['g_type'], dqn.cfg['NUM_MIN'], dqn.cfg['NUM_MAX'])
    vc_list = []
    iterations = []
    for line in open(VCFile):
        iteration, approx_ratio = line.split(' ')
        try:
            vc_list.append(float(approx_ratio))
            iterations.append(int(iteration))
        except:
            continue
    min_arg_vc = np.argmin(vc_list)
    min_vc = str(np.round(np.min(vc_list), 6))
    best_model_iter = iterations[min_arg_vc]

    best_model = './logs/%s/nrange_%d_%d_iter_%d.ckpt' % (dqn.cfg['g_type'], dqn.cfg['NUM_MIN'], dqn.cfg['NUM_MAX'], best_model_iter)
    print(best_model)
    return best_model, VCFile, min_vc

def find_all_models(dqn, valid_file_path):
    g_type = dqn.cfg['g_type']
    NUM_MIN = dqn.cfg['NUM_MIN']
    NUM_MAX = dqn.cfg['NUM_MAX']
    if valid_file_path:
        VCFile = valid_file_path
    else:
        VCFile = './logs/%s/ModelVC_%d_%d.csv'%(dqn.cfg['g_type'], dqn.cfg['NUM_MIN'], dqn.cfg['NUM_MAX'])
    vc_list = []
    iterations = []
    model_list = []
    for line in open(VCFile):
        iteration, approx_ratio = line.split(' ')
        vc_list.append(float(approx_ratio))
        iterations.append(int(iteration))
        model_list.append(f'./logs/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}_iter_{iteration}.ckpt')
    min_arg_vc = np.argmin(vc_list)
    min_vc = str(np.round(np.min(vc_list), 6))
    best_model_iter = iterations[min_arg_vc]

    best_model = f'./logs/{g_type}/nrange_{NUM_MIN}_{NUM_MAX}_iter_{best_model_iter}.ckpt'
    print(best_model)
    return model_list, best_model, VCFile, min_vc

def copy_multiple_files(origin, file_names, target, name_addon):
    try:
        os.makedirs(target)
    except:
        pass
    for file_name in file_names:
        new_file_name = file_name.split('.')
        new_file_name[-3] = new_file_name[-3] + name_addon
        new_file_name = '.'.join(new_file_name)
        copy(origin + file_name, target + new_file_name)

def save_best_model(dqn, config_path):
    g_type = dqn.cfg['g_type']
    NUM_MIN = dqn.cfg['NUM_MIN']
    NUM_MAX = dqn.cfg['NUM_MAX']
    print("Searching for best model...")
    vcfile_path = None
    model_list, best_model, vcfile_path, min_tour_length = find_all_models(dqn, valid_file_path=vcfile_path)
    min_tour_length = ''.join(min_tour_length.split('.'))
    model_files = []
    
    print("Saving best model...")
    best_file_name = best_model.split('/')[-1]
    
    # base path to the model files and the vcmodel file
    base_path = '/'.join(vcfile_path.split('/')[0:-1]) + '/'
    files = os.listdir(base_path)
    checkpoint_suffixes = ['.data-00000-of-00001', '.index', '.meta']
    # extract all checkpoint files
    best_model_files = [best_file_name+suffix for suffix in checkpoint_suffixes]

    node_range = '_'.join(best_file_name.split('_')[0:3])
    target = './best_models/{}/{}/'.format(g_type, node_range + '_len_' + min_tour_length)
    copy_multiple_files(origin=base_path, file_names=best_model_files, target=target, name_addon='_len_'+str(min_tour_length))

    print("Saving VCFile...")
    # modify only ne name not the file ending..
    vcfile_name = vcfile_path.split('/')[-1].split('.') 
    vcfile_name[-2] = vcfile_name[-2] + '_len_' + min_tour_length
    vcfile_name = '.'.join(vcfile_name)
    copy(vcfile_path, target + vcfile_name)

    print("Saving Loss file...")
    loss_file_name = f'Loss_{NUM_MIN}_{NUM_MAX}.csv'
    copy(base_path+loss_file_name, target+loss_file_name)

    print("Saving config file...")
    copy(config_path, target + config_path.split('/')[-1].split('_')[-1])

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