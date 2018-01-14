# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import sys
import io
import math
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
    print("epoch \t steps \t train_loss \t lr \t time")
    prog = general_utils.Progbar(target=iter_per_epoch, verbose=2)
    time_s = time()

    for epoch in range(args.epoch):
        random.shuffle(train_data)
        train_iter = data.iterator.pool(train_data, args.batchsize,
                                        key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                        random_shuffler=data.iterator.RandomShuffler())

        for num_steps, train_batch in tqdm(enumerate(train_iter)):
            model.train()
            optimizer.zero_grad()

            # ---------- One iteration of the training loop ----------
            # train_batch = next(iter(train_iter))
            in_arrays = utils.seq2seq_pad_concat_convert(train_batch, -1)
            loss, acc_stat, perp = model(*in_arrays)

            loss.backward()

            if args.use_fixed_lr:
                norm = torch.nn.utils.clip_grad_norm(model.parameters(), args.max_norm)
            optimizer.step()

            if args.debug:
                dummy_norm = 50.0
                norm = torch.nn.utils.clip_grad_norm(model.parameters(), dummy_norm)
                prog.update(num_steps, values=[("train loss", loss.data.cpu().numpy()[0]),], exact=[("norm", norm)])

        # Check the validation accuracy of prediction after every epoch
        test_losses = []
        test_iter = data.iterator.pool(valid_data, args.batchsize // 4,
                                       key=lambda x: data.utils.interleave_keys(len(x[0]), len(x[1])),
                                       random_shuffler=data.iterator.RandomShuffler())

        count, total = 0, 0
        for test_batch in test_iter:
            model.eval()
            in_arrays = utils.seq2seq_pad_concat_convert(test_batch, -1)
            loss_test, acc_stat, perp = model(*in_arrays)
            count += acc_stat[0]
            total += acc_stat[1]
            test_losses.append(loss_test.data.cpu().numpy())

        accuracy = (count / total).cpu().tolist()[0]
        print('val_loss:{:.04f} \t Acc:{:.04f} \t time: {:.2f}'.format(np.mean(test_losses), accuracy,
                                                                       time()-time_s))

        if not args.no_bleu:
            if args.beam_size > 1 and epoch > 30:
                CalculateBleu(model, valid_data, 'val/main/bleu', batch=1, beam_size=args.beam_size)()
            else:
                CalculateBleu(model, valid_data, 'val/main/bleu', batch=args.batchsize // 4)()


if __name__ == '__main__':
    main()
