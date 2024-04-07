import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=500, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: m")
    parser.add_argument('--batch_size', type=int, default=20, help="local batch size: B")
    parser.add_argument('--test_bs', type=int, default=768, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--c_lr', type=float, default=2, help="cluster learning rate")
    parser.add_argument('--g_lr', type=float, default=2, help="master learning rate")

    # model arguments
    parser.add_argument('--model', type=str, default='cnn', help='model name')
    parser.add_argument('--resume', type=str, default='', help='resume model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=1, help="number of channels of imges")
    parser.add_argument('--gpu', type=str, default='1', help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--setting', type=str, required=True, help='grouping experiment settings')

    # data distribution patterns
    parser.add_argument('--inter_noniid', default=3, type=int, help='non i.i.d. pattern among group (False value indicates i.i.d.)')
    parser.add_argument('--intra_noniid', default='1,1,1,1', type=str, help='non i.i.d. pattern within each group (False value indicates i.i.d.)')
    parser.add_argument('--replacement', type=bool, default=False, help='if sampling with replacement')

    parser.add_argument('--contrast', action='store_true', help='if False, instantiate as a normal HFL, \
        otherwise, instantiate as stadard FL for contrast, with the same data distribution, \
        !!! this is achieved by sum up `--groups_pattern` and `--sampling_nums`, and overwrite `--cluster_periods` with `--master_period`')

    # group arguements
    parser.add_argument('--groups_pattern', type=str, default='25,25,25,25', help="num of users each group: m_i")
    parser.add_argument('--sampling_nums', type=str, default='5,5,5,5', help="num of sampling subset of users: n_i")
    parser.add_argument('--log_freq', type=int, default=1, help="number of rounds before computing loss")
    parser.add_argument('--cluster_periods', type=str, default='10,10,10,10',help='iterations before cluster aggregation: I_i')
    parser.add_argument('--master_period', type=int, default=100, help='iterations before master aggregation: G')

    parser.add_argument('--num_dataset_workers', type=int, default=0, help="number of workers when loading data")
    
    args = parser.parse_args()
    return args
