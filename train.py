# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import io
import itertools
import numpy as np
import random
from time import time
import torch
from tqdm import tqdm

import evaluator
import net
import optimizer as optim
from torchtext import data
import utils
import general_utils
from config import get_train_args


def report_func(epoch, batch, num_batches, start_time, report_stats, report_every):
    """
    This is the user-defined batch-level training progress
    report function.
    Args:
        epoch(int): current epoch count.
        batch(int): current batch count.
        num_batches(int): total number of batches.
        start_time(float): last report time.
        lr(float): current learning rate.
        report_stats(Statistics): old Statistics instance.
    Returns:
        report_stats(Statistics): updated Statistics instance.
    """
    if batch % report_every == -1 % report_every:
        report_stats.output(epoch, batch+1, num_batches, start_time)
        report_stats = utils.Statistics()

    return report_stats


class CalculateBleu(object):
    def __init__(self, model, test_data, key, batch=50, max_length=50, beam_size=1):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.batch = batch
        self.device = -1
        self.max_length = max_length
        self.beam_size = beam_size

    def __call__(self):
        self.model.eval()
        references = []
        hypotheses = []
        for i in tqdm(range(0, len(self.test_data), self.batch)):
            sources, targets = zip(*self.test_data[i:i + self.batch])
            references.extend(t.tolist() for t in targets)
            if self.beam_size > 1:
                ys = [self.model.translate(sources, self.max_length, beam=2)]
            else:
                ys = [y.tolist() for y in self.model.translate(sources, self.max_length, beam=False)]
            hypotheses.extend(ys)
        bleu = evaluator.BLEUEvaluator().evaluate(references, hypotheses)
        print('BLEU:', bleu.score_str())
        print('')


def main():
    args = get_train_args()
    print(json.dumps(args.__dict__, indent=4))

    # Reading the int indexed text dataset
    train_data = np.load(os.path.join(args.input, args.save_data + ".train.npy")).tolist()
    valid_data = np.load(os.path.join(args.input, args.save_data + ".valid.npy")).tolist()
    test_data = np.load(os.path.join(args.input, args.save_data + ".test.npy")).tolist()

    # Reading the vocab file
    with io.open(os.path.join(args.input, args.save_data + '.vocab.src.json'), encoding='utf-8') as f:
        source_id2w = json.load(f, cls=utils.Decoder)

    with io.open(os.path.join(args.input, args.save_data + '.vocab.trg.json'), encoding='utf-8') as f:
        target_id2w = json.load(f, cls=utils.Decoder)

    # Define Model
    model = net.Transformer(args.layer,
                            len(source_id2w),
                            len(target_id2w),
                            args.unit,
                            multi_heads=args.multi_heads,
                            dropout=args.dropout,
                            max_length=500,
                            label_smoothing=args.label_smoothing,
                            embed_position=args.embed_position,
                            layer_norm=True,
                            tied=args.tied,
                            pos_attention=args.pos_attention)

    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    if not args.use_fixed_lr:
        optimizer = optim.TransformerAdamTrainer(model)
    else:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)
    time_s = time()

    for epoch in range(args.epoch):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data, args.batchsize,
                                        key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                        random_shuffler=data.iterator.RandomShuffler())
        report_stats = utils.Statistics()
        train_stats = utils.Statistics()
        valid_stats = utils.Statistics()

        for num_steps, train_batch in enumerate(train_iter):
            model.train()
            optimizer.zero_grad()

            # ---------- One iteration of the training loop ----------
            src_words = len(list(itertools.chain.from_iterable(list(zip(*train_batch))[0])))
            report_stats.n_src_words += src_words
            train_stats.n_src_words += src_words

            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            loss, stat = model(*in_arrays)
            loss.backward()

            if args.use_fixed_lr:
                norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            optimizer.step()

            report_stats.update(stat)
            train_stats.update(stat)
            report_stats = report_func(epoch, num_steps, iter_per_epoch, time_s, report_stats, args.report_every)

        # Check the validation accuracy of prediction after every epoch
        test_iter = data.iterator.pool(valid_data, args.batchsize // 4,
                                       key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                       random_shuffler=data.iterator.RandomShuffler())

        for test_batch in test_iter:
            model.eval()
            in_arrays = utils.seq2seq_pad_concat_convert(test_batch, -1)
            loss_test, stat = model(*in_arrays)
            valid_stats.update(stat)

        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())

        # 2. Validate on the validation set.
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())

        # print('val_loss:{:.04f} \t Acc:{:.04f} \t time: {:.2f}'.format(np.mean(test_losses), accuracy,
        #                                                                time()-time_s))

        if not args.no_bleu:
            if args.beam_size > 1 and epoch > 30:
                CalculateBleu(model, valid_data, 'val/main/bleu', batch=1, beam_size=args.beam_size)()
            else:
                CalculateBleu(model, valid_data, 'val/main/bleu', batch=args.batchsize // 4)()


if __name__ == '__main__':
    main()
