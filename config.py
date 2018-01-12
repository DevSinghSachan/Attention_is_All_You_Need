import os
from argparse import ArgumentParser
from datetime import datetime
import random


def get_args():
    parser = ArgumentParser(description='Implementation of "Attention is All You Need" in Pytorch')

    parser.add_argument('--batchsize', '-b', type=int, default=10,
                        help='Number of sentences in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=40,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--unit', '-u', type=int, default=512,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers')
    parser.add_argument('--multi_heads', type=int, default=8,
                        help='Number of heads in attention mechanism')
    parser.add_argument('--dropout', '-d', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--tied', dest='tied', action='store_true',
                        help='tie target word embedding and output softmax layer')
    parser.set_defaults(tied=False)

    parser.add_argument('--beam_size', dest='beam_size', type=int, default=1,
                        help='Beam size during translation')
    parser.add_argument('--input', '-i', type=str, default='./data/ja_en',
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
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--source-vocab', type=int, default=40000,
                        help='Vocabulary size of source language')
    parser.add_argument('--target-vocab', type=int, default=40000,
                        help='Vocabulary size of target language')
    parser.add_argument('--no_bleu', dest='-no_bleu', action='store_true',
                        help='Skip BLEU calculation')
    parser.set_defaults(no_bleu=True)

    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Use label smoothing for cross-entropy')
    parser.add_argument('--embed-position', action='store_true',
                        help='Use position embedding rather than sinusoid')

    parser.add_argument('--use_fixed_lr', dest='use_fixed_lr' ,action='store_true',
                        help='Use fixed learning rate rather than the ' +
                             'annealing proposed in the paper')
    parser.set_defaults(use_fixed_lr=False)

    # In debug, print progress bar, otherwise not
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    args = parser.parse_args()
    return args

get_args()
