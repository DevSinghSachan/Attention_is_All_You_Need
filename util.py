import os
from argparse import ArgumentParser
from datetime import datetime
import random

def get_args():
    parser = ArgumentParser(description='Implementation of "Attention is All You Need" in Pytorch')

    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--head', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--input', '-i', type=str, default='./data',
                        help='Input directory')
    parser.add_argument('--source', '-s', type=str,
                        default='train.ja',
                        help='Filename of train data for source language')
    parser.add_argument('--target', '-t', type=str,
                        default='train.en',
                        help='Filename of train data for target language')
    parser.add_argument('--source-valid', '-svalid', type=str,
                        default='dev.ja',
                        help='Filename of validation data for source language')
    parser.add_argument('--target-valid', '-tvalid', type=str,
                        default='dev.en',
                        help='Filename of validation data for target language')
    parser.add_argument('--target-valid-raw', '-tvalid_raw', type=str,
                        default='dev.en',
                        help='Filename of raw validation data for target language')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--no-bleu', '-no-bleu', action='store_true',
                        help='Skip BLEU calculation')
    parser.add_argument('--use-label-smoothing', action='store_true',
                        help='Use label smoothing for cross entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')
    parser.add_argument('--use-fixed-lr', action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')
    parser.add_argument('--hyp_dev', default='hyp_dev.txt')
    parser.add_argument('--hyp_test', default='hyp_test.txt')

    # In debug, print progress bar, otherwise not
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    return args

get_args()
