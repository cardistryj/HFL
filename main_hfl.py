import gc
import json
import os
import copy
import numpy as np
import logging
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict

from utils.sampling import partition
from utils.options import args_parser
from models.update import LocalUpdate
from models.networks import MLP, CNN
from models.test import Tester

from multiprocessing import pool

# parse args
args = args_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu 
args.device = 'cuda' if torch.cuda.is_available() and args.gpu != '-1' else 'cpu'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

root_dir = os.path.join('output', args.dataset, args.model)
# time_now = time.strftime('%Y%m%d%H%M%S', time.localtime())

target_args = 'num_users batch_size lr c_lr g_lr groups_pattern \
sampling_nums log_freq cluster_periods master_period replacement inter_noniid intra_noniid contrast'.split(' ')
def args_filter(arg):
    return not arg.startswith('_') and arg in target_args

hyperparams = {}
for hp in filter(args_filter, dir(args)):
    hyperparams[hp] = eval(f'args.{hp}')

params_str = json.dumps(hyperparams, indent=4)
fed_config_str = str(hash(params_str))

exp_config_dir = os.path.join(root_dir, args.setting, fed_config_str)
exp_instance_dir = os.path.join(exp_config_dir, f'seed-{args.seed}')
if not os.path.exists(exp_instance_dir):
    os.makedirs(exp_instance_dir)

# output config file for hyperparameters
config_path = os.path.join(exp_config_dir, 'config.json')
if not os.path.exists(config_path):
    with open(config_path, 'w') as f:
        f.write(params_str)

log_path = os.path.join(exp_instance_dir, 'output.log')
logger = logging.getLogger(__name__)
logging.basicConfig(
    format='[%(asctime)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG,
    handlers=[
        logging.FileHandler(log_path, mode = 'a' if args.resume else 'w'),
        logging.StreamHandler()
    ])

#group params
cluster_user_nums = list(map(int, args.groups_pattern.split(',')))
num_sampled_clients = list(map(int, args.sampling_nums.split(',')))
assert sum(cluster_user_nums) == args.num_users
assert len(list(filter(lambda x: x[0] < x[1], zip(cluster_user_nums, num_sampled_clients)))) == 0, 'n_i larger than m_i'

# load dataset and split users
normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
if args.dataset == 'mnist':
    trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
else:
    exit('Error: unrecognized dataset')
img_size = dataset_train[0][0].shape

inter_noniid = args.inter_noniid
intra_noniid = list(map(int, args.intra_noniid.split(',')))
if inter_noniid:
    # group noniid is only available in uniform grouping case
    assert all([num == cluster_user_nums[0] for num in cluster_user_nums])

labels = np.array(dataset_train.targets)
cluster_all_indices = partition(labels, cluster_user_nums, inter_noniid, np.arange(len(labels))).values()

dict_users = {}
offset = 0
for indice, num, noniid in zip(cluster_all_indices, cluster_user_nums, intra_noniid):
    cluster_user_dict = partition(labels[indice], num, noniid, indice, offset)
    offset += num
    dict_users.update(cluster_user_dict)

if args.contrast:
    # This transform the HFL instance into standard FL for contrast
    # sum up `--groups_pattern` and `--sampling_nums`, and overwrite `--cluster_periods` with `--master_period`
    cluster_user_nums = [sum(cluster_user_nums)]
    num_sampled_clients = [sum(num_sampled_clients)]
    args.cluster_periods = str(args.master_period)

num_samples_users = np.array([len(dict_users[i]) for i in dict_users])
cluster_user_indices = []
cluster_weights = []

#lweight = []
count = 0
for idx in range(len(cluster_user_nums)):        
    cluster_user_indices.append(np.arange(count, count + cluster_user_nums[idx]))
    cluster_weights.append(np.sum(num_samples_users[count:count + cluster_user_nums[idx]])/np.sum(num_samples_users))
    count += cluster_user_nums[idx]

cluster_periods = list(map(int, args.cluster_periods.split(',')))
assert len(list(filter(lambda x: args.master_period % x, cluster_periods))) == 0, 'G is not a multiple of I'
# G/I
omegas = list(map(lambda x: args.master_period // x, cluster_periods))

if args.model == 'cnn' and args.dataset == 'mnist':
    net_master = CNN(args=args).to(args.device)
elif args.model == 'mlp':
    len_in = 1
    for x in img_size:
        len_in *= x
    net_master = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes)
    net_master = nn.DataParallel(net_master)
    net_master = net_master.to(args.device)
else:
    exit('Error: unrecognized model')

start_epochs = 0
if args.resume:
    start_epochs = int(args.resume)
    resume_model_path = os.path.join(exp_instance_dir, f'{start_epochs}.pth')
    logger.info(f'resuming model from epoch {start_epochs}')
    net_master.load_state_dict(torch.load(resume_model_path))
else:
    logger.info(net_master)
net_master.train() #net initialization

def cluster_round(cluster_idx):
    net_cluster = copy.deepcopy(net_master).to(args.device)
    for i in range(0,omegas[cluster_idx]):
        sampled_clients = np.array([])
        sampled_clients = np.random.choice(cluster_user_indices[cluster_idx],size=num_sampled_clients[cluster_idx],replace=args.replacement)

        lweight = num_samples_users[sampled_clients]
        lweight = lweight/np.sum(lweight)

        cluster_state = OrderedDict()

        for j in range(0, len(sampled_clients)):
            idx = sampled_clients[j]

            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx], iters = cluster_periods[cluster_idx], nums=num_samples_users[idx])
            net_local = copy.deepcopy(net_cluster).to(args.device)
            w = local.train(net=net_local)

            for k in w.keys():
                cluster_state[k] = cluster_state.get(k, 0) + w[k].detach() * lweight[j]

            del net_local
            gc.collect()

        # applying cluster learning rate
        for k in cluster_state.keys():
            cluster_state[k] = net_cluster.state_dict()[k] + args.c_lr * (cluster_state[k] - net_cluster.state_dict()[k])

        net_cluster.load_state_dict(cluster_state)

    return cluster_state

def aggregate_state(model_state_list, coefficient):
    assert len(model_state_list) == len(coefficient)
    avg_model_state = OrderedDict()
    for local_model_state, coef in zip(model_state_list, coefficient):
        for key in local_model_state.keys():
            avg_model_state[key] = avg_model_state.get(key, 0) + coef * local_model_state[key]
    
    return avg_model_state

tester = Tester(args)

for iters in range(start_epochs, start_epochs + args.epochs):
    with pool.ThreadPool(processes=3) as workhorse:
        cluster_states = workhorse.map(cluster_round, range(len(cluster_user_nums)))

    master_state = aggregate_state(cluster_states, cluster_weights)

    # applying master learning rate
    for k in master_state.keys():
        master_state[k] = net_master.state_dict()[k] + args.g_lr * (master_state[k] - net_master.state_dict()[k])

    net_master.load_state_dict(master_state)

    # compute training/test accuracy/loss
    if (iters+1) % args.log_freq == 0:
        tester.set_model(net_master)
        with torch.no_grad():
            acc_test, loss_test = tester.run_test(dataset_test)
        logger.info(f'Round {iters}, test loss {loss_test:.3f}, test acc {acc_test:.3f}')

model_path = os.path.join(exp_instance_dir, f'{start_epochs + args.epochs}.pth')
torch.save(net_master.state_dict(), model_path)