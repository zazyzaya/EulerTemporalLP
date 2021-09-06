from argparse import ArgumentParser

import pandas as pd

import loaders.load_lanl as lanl
import loaders.load_pico as pico
import loaders.load_optc as optc
from models.recurrent import GRU, LSTM, Lin, EmptyModel
from models.embedders import \
    static_gcn_rref, static_gat_rref, static_sage_rref, \
    dynamic_gcn_rref, dynamic_gat_rref, dynamic_sage_rref 

from models.tedge import tedge_rref
from spinup import run_all

DEFAULT_TR = {
    'lr': 0.01,
    'epochs': 500,
    'min': 1,
    'patience': 10,
    'nratio': 10,
    'val_nratio': 1,
    'decompress': True
}

HOME = '/mnt/raid0_24TB/isaiah/code/EulerTemporalLP/'
def get_args():
    global DEFAULT_TR

    ap = ArgumentParser()

    ap.add_argument(
        '-d', '--delta',
        type=float, default=2.0
    )

    ap.add_argument(
        '-w', '--workers',
        type=int, default=4
    )

    ap.add_argument(
        '-T', '--threads',
        type=int, default=1
    )

    ap.add_argument(
        '-e', '--encoder',
        choices=['GCN', 'GAT', 'SAGE', 'GIN'],
        type=str.upper,
        default="GCN"
    )

    ap.add_argument(
        '-r', '--rnn',
        choices=['GRU', 'LSTM', 'NONE', 'MLP'],
        type=str.upper,
        default="GRU"
    )

    ap.add_argument(
        '-H', '--hidden',
        type=int,
        default=32
    )

    ap.add_argument(
        '-z', '--zdim',
        type=int,
        default=16
    )

    ap.add_argument(
        '-n', '--ngrus',
        type=int,
        default=1
    )

    ap.add_argument(
        '-t', '--tests',
        type=int, 
        default=1
    )

    ap.add_argument(
        '-l', '--load',
        action='store_true'
    )

    ap.add_argument(
        '--fpweight',
        type=float,
        default=0.6
    )

    ap.add_argument(
        '--nowrite',
        action='store_true'
    )

    ap.add_argument(
        '--impl', '-i',
        type=str.upper,
        choices=['S', 'STATIC', 'D', 'DYNAMIC', 'T', 'TEDGE']
    )

    ap.add_argument(
        '--dataset',
        default='LANL', 
        type=str.upper,
        choices=['LANL', 'L', 'PICO', 'P', 'OPTC', 'O']
    )

    ap.add_argument(
        '--compressed',
        action='store_false'
    )

    args = ap.parse_args()
    args.te_end = None
    assert args.fpweight >= 0 and args.fpweight <=1, '--fpweight must be a value between 0 and 1 (inclusive)'

    readable = str(args)
    print(readable)

    impl = args.impl[0]
    args.static = impl
    DEFAULT_TR['decompress'] = args.compressed
    
    # Parse dataset info 
    if args.dataset.startswith('L'):
        args.loader = lanl.load_lanl_dist
        args.tr_start = 0
        args.tr_end = lanl.DATE_OF_EVIL_LANL
        args.val_times = None
        args.te_times = [(args.tr_end, lanl.TIMES['all'])]
        args.delta = int(args.delta * (60**2))
        args.manual = False 

    elif args.dataset.startswith('P'):
        args.loader = pico.load_pico
        args.tr_start = pico.PICO_START
        args.tr_end = pico.DATE_OF_EVIL_PICO
        args.val_times = None
        args.te_times = [(args.tr_end, pico.PICO_END)]
        args.delta = int(args.delta * (60**2))
        args.manual = True

    elif args.dataset.startswith('O'):
        args.loader = optc.load_optc_dist
        args.tr_start = 0 #optc.TIMES['val_start']
        args.tr_end = optc.TIMES['val_end']
        args.val_times = None #(optc.TIMES['val_start'], optc.TIMES['val_end'])
        args.te_times = [optc.DAY1, optc.DAY2, optc.DAY3, optc.ALL]
        args.delta = int(args.delta * 60)
        args.manual = False 

    # The checking in argparse should never allow it to
    # reach this else block, but just in case
    else:
        print("Dataset %s not implimented" % args.dataset)
        exit()

    # Convert from str to function pointer
    if args.encoder == 'GCN':
        args.encoder = static_gcn_rref if impl == 'S' \
            else dynamic_gcn_rref
    elif args.encoder == 'GAT':
        args.encoder = static_gat_rref if impl == 'S' \
            else dynamic_gat_rref
    else:
        args.encoder = static_sage_rref if impl == 'S' \
            else dynamic_sage_rref

    if impl == 'T':
        args.encoder = tedge_rref

    if args.rnn == 'GRU':
        args.rnn = GRU
    elif args.rnn == 'LSTM':
        args.rnn = LSTM 
    elif args.rnn == 'MLP':
        args.rnn = Lin
    else:
        args.rnn = EmptyModel

    return args, readable

if __name__ == '__main__':
    args, argstr = get_args() 

    if args.rnn != EmptyModel:
        worker_args = [args.hidden, args.hidden]
        rnn_args = [args.hidden, args.hidden, args.zdim]
    else:
        # Need to tell workers to output in embed dim
        worker_args = [args.hidden, args.zdim]
        rnn_args = [args.hidden, args.hidden, args.zdim]

    stats = [
        run_all(
            args.workers, 
            args.rnn, 
            rnn_args,
            args.encoder, 
            worker_args, 
            args.delta,
            args.load,
            args.fpweight,
            args.static,
            args.loader, 
            args.tr_start,
            args.tr_end, 
            args.val_times,
            args.te_times,
            args.manual,
            DEFAULT_TR
        )
        for _ in range(args.tests)
    ]

    # Don't write out if nowrite
    if args.nowrite:
        exit() 

    f = open(HOME+'results/stats.txt', 'a')
    f.write(str(argstr) + '\n\n')

    dfs = [pd.DataFrame(s) for s in list(zip(*stats))]
    for i in range(len(dfs)):
        df = dfs[i]

        compressed = pd.DataFrame(
            [df.mean(), df.sem()],
            index=['mean', 'stderr']
        ).to_csv().replace(',', ', ')

        full = df.to_csv(index=False, header=False)
        full = full.replace(',', ', ')

        f.write('[%d]:\n' % i)        
        f.write(str(compressed) + '\n')
        f.write(full + '\n')

    f.close()