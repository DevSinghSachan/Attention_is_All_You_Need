# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os.path
from nltk.translate import bleu_score
import numpy as np
import six
import random
import io
import subprocess
from time import time

import torch
from torch.autograd import Variable
import chainer
from chainer.dataset import convert

import preprocess
import net
from subfuncs import TransformerAdamTrainer
from torchtext import data
import general_utils
from config import get_args

if torch.cuda.is_available():
    torch.cuda.set_device(0)


def seq2seq_pad_concat_convert(xy_batch, device, eos_id=1, bos_id=3):
    """
    Args:
        xy_batch (list of tuple of two numpy.ndarray-s or cupy.ndarray-s):
            xy_batch[i][0] is an array
            of token ids of i-th input sentence in a minibatch.
            xy_batch[i][1] is an array
            of token ids of i-th target sentence in a minibatch.
            The shape of each array is `(sentence length, )`.
        device (int or None): Device ID to which an array is sent. If it is
            negative value, an array is sent to CPU. If it is positive, an
            array is sent to GPU with the given ID. If it is ``None``, an
            array is left in the original device.

    Returns:
        Tuple of Converted array.
            (input_sent_batch_array, target_sent_batch_input_array,
            target_sent_batch_output_array).
            The shape of each array is `(batchsize, max_sentence_length)`.
            All sentences are padded with -1 to reach max_sentence_length.
    """

    x_seqs, y_seqs = zip(*xy_batch)
    x_block = convert.concat_examples(x_seqs, device, padding=0)
    y_block = convert.concat_examples(y_seqs, device, padding=0)

    # Add EOS
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)

    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    x_block = np.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)

    y_out_block = np.pad(y_block, ((0, 0), (0, 1)), 'constant', constant_values=0)

    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = eos_id

    y_in_block = np.pad(y_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)

    # Converting from numpy format to Torch Tensor
    x_block, y_in_block, y_out_block = Variable(torch.LongTensor(x_block)), \
                                       Variable(torch.LongTensor(y_in_block)), \
                                       Variable(torch.LongTensor(y_out_block))

    if torch.cuda.is_available():
        x_block, y_in_block, y_out_block = x_block.cuda(), y_in_block.cuda(), y_out_block.cuda()

    return x_block, y_in_block, y_out_block


def source_pad_concat_convert(x_seqs, device, eos_id=1, bos_id=3):
    x_block = convert.concat_examples(x_seqs, device, padding=0)

    # add eos
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    x_block = np.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)
    return x_block


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_length = max_length

    def __call__(self):
        self.model.eval()
        references = []
        hypotheses = []
        for i in range(0, len(self.test_data), self.batch):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend([[t.tolist()] for t in targets])

            sources = [chainer.dataset.to_device(self.device, x) for x in sources]

            ys = [y.tolist() for y in self.model.translate(sources, self.max_length, beam=False)]
            # greedy generation for efficiency
            hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(references, hypotheses, smoothing_function=bleu_score.SmoothingFunction().method1)
        print('BLEU:', bleu)


def main():
    args = get_args()

    print(json.dumps(args.__dict__, indent=4))

    # Check file
    source_path = os.path.join(args.input, args.source)
    source_vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + preprocess.count_words(source_path, args.source_vocab)
    source_data = preprocess.make_dataset(source_path, source_vocab)

    target_path = os.path.join(args.input, args.target)
    target_vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + preprocess.count_words(target_path, args.target_vocab)
    target_data = preprocess.make_dataset(target_path, target_vocab)
    assert len(source_data) == len(target_data)

    print("Source Vocab: {}".format(len(source_vocab)))
    print("Target Vocab: {}".format(len(target_vocab)))

    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data) if 0 < len(s) < 50 and 0 < len(t) < 50]
    print('Filtered training data size: %d' % len(train_data))

    source_path = os.path.join(args.input, args.source_valid)
    source_data = preprocess.make_dataset(source_path, source_vocab)
    target_path = os.path.join(args.input, args.target_valid)
    target_data = preprocess.make_dataset(target_path, target_vocab)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data) if 0 < len(s) and 0 < len(t)]

    hyp_dev_path = os.path.join(args.input, args.hyp_dev)
    hyp_test_path = os.path.join(args.input, args.hyp_test)
    ref_dev_path = os.path.join(args.input, args.target_valid_raw)

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net.Transformer(args.layer,
                            min(len(source_ids), len(source_words)),
                            min(len(target_ids), len(target_words)),
                            args.unit,
                            h=args.head,
                            dropout=args.dropout,
                            max_length=500,
                            label_smoothing=args.label_smoothing,
                            embed_position=args.embed_position,
                            layer_norm=True,
                            tied=args.tied)

    if args.gpu >= 0:
        model.cuda(args.gpu)

    print(model)

    # Setup Optimizer
    optimizer = TransformerAdamTrainer(model)
    # train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    # test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize // 2, repeat=False, shuffle=False)

    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)
    print("epoch \t steps \t train_loss \t lr \t time")
    prog = general_utils.Progbar(target=iter_per_epoch)
    time_s = time()

    for epoch in range(args.epoch):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data, args.batchsize,
                                        key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                        random_shuffler=data.iterator.RandomShuffler())

        for num_steps, train_batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()

            # ---------- One iteration of the training loop ----------
            # train_batch = next(iter(train_iter))
            in_arrays = seq2seq_pad_concat_convert(train_batch, -1)
            loss, acc, perp = model(*in_arrays)

            loss.backward()
            # norm = torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
            optimizer.step()

            if args.debug:
                dummy_norm = 50.0
                norm = torch.nn.utils.clip_grad_norm(model.parameters(), dummy_norm)
                prog.update(num_steps, values=[("train loss", loss.data.cpu().numpy()[0]),], exact=[("norm", norm)])

        # Check the validation accuracy of prediction after every epoch
        prog = general_utils.Progbar(target=iter_per_epoch)
        test_losses = []
        test_iter = data.iterator.pool(test_data, args.batchsize,
                                       key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                       random_shuffler=data.iterator.RandomShuffler())

        for test_batch in test_iter:
            model.eval()
            in_arrays = seq2seq_pad_concat_convert(test_batch, -1)
            loss_test, acc, perp = model(*in_arrays)
            test_losses.append(loss_test.data.cpu().numpy())

        print('val_loss:{:.04f} \t time: {:.2f}'.format(np.mean(test_losses), time()-time_s))
        CalculateBleu(model, test_data, 'val/main/bleu', batch=args.batchsize//4)()


if __name__ == '__main__':
    main()
