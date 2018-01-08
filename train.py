# encoding: utf-8
from __future__ import unicode_literals, print_function

import argparse
import json
import os.path
from nltk.translate import bleu_score
import numpy as np
import six
import io
import subprocess
from time import time

import torch
from torch.autograd import Variable

import chainer
from chainer import cuda
from chainer.dataset import convert
from chainer import reporter
from chainer import training
from chainer.training import extensions

import preprocess
import net
from subfuncs import VaswaniRule


def postprocess(file_, file2_, vocab, hypotheses):
    # Save the Hypothesis to output file
    with io.open(file_, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')

    # Detokenize the Output
    command = 'detokenize < {} > {}.detok'.format(file_, file_)
    subprocess.check_call(command, shell=True)

    # Compute the BLEU Score (uselower case)
    command = 'perl multi-bleu.perl {} < {}.detok'.format(file2_, file_)
    subprocess.check_call(command, shell=True)


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

    # The paper did not mention eos
    # add eos

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
    xp = cuda.get_array_module(x_block)

    # add eos
    with xp.cuda.Device(device):
        x_block = xp.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)

        for i_batch, seq in enumerate(x_seqs):
            x_block[i_batch, len(seq)] = eos_id

        x_block = xp.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)

        return x_block


class CalculateBleu(chainer.training.Extension):
    trigger = 1, 'epoch'
    priority = chainer.training.PRIORITY_WRITER

    def __init__(self, model, test_data, key, dev_hyp_file, target_vocab, dev_ref_file, batch=50, device=-1,
                 max_length=50):
        self.model = model
        self.test_data = test_data
        self.key = key
        self.dev_hyp_file = dev_hyp_file
        self.dev_ref_file = dev_ref_file
        self.target_vocab = target_vocab
        self.batch = batch
        self.device = device
        self.max_length = max_length

    def __call__(self, trainer):
        print('## Calculate BLEU')
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                references = []
                hypotheses = []
                for i in range(0, len(self.test_data), self.batch):
                    sources, targets = zip(*self.test_data[i:i + self.batch])
                    references.extend([[t.tolist()] for t in targets])

                    sources = [chainer.dataset.to_device(self.device, x) for x in sources]

                    ys = [y.tolist() for y in self.model.translate(sources, self.max_length, beam=False)]
                    # greedy generation for efficiency
                    hypotheses.extend(ys)

        bleu = bleu_score.corpus_bleu(references, hypotheses,
                                      smoothing_function=bleu_score.SmoothingFunction().method0) * 100

        print('BLEU:', bleu)
        reporter.report({self.key: bleu})
        postprocess(self.dev_hyp_file, self.dev_ref_file, self.target_vocab, hypotheses)


def main():
    parser = argparse.ArgumentParser(description='Implementation of "Attention is All You Need"')

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
    args = parser.parse_args()

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
                            use_label_smoothing=args.use_label_smoothing,
                            embed_position=args.embed_position)

    if args.gpu >= 0:
        model.cuda(args.gpu)

    # Setup Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # optimizer = chainer.optimizers.Adam(alpha=5e-5, beta1=0.9, beta2=0.98, eps=1e-9)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)

    test_iter = chainer.iterators.SerialIterator(test_data, args.batchsize,
                                                 repeat=False, shuffle=False)

    iter_per_epoch = len(train_data) // args.batchsize
    print('Number of iter/epoch =', iter_per_epoch)
    print("epoch \t steps \t train_loss \t lr \t time")

    num_steps = 0
    time_s = time()

    model.train()
    while train_iter.epoch < args.epoch:
        optimizer.zero_grad()

        num_steps += 1

        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        in_arrays = seq2seq_pad_concat_convert(train_batch, -1)
        # model.set_dropout(args.dropout)
        loss = model(*in_arrays)

        loss.backward()
        optimizer.step()

        if num_steps % 200 == 0:
            print("{:.03f}/{:02d} \t {}\t {:.04f}\t {:.05f}\t {:.01f} sec".format(train_iter.epoch_detail,
                                                                                  train_iter.epoch + 1,
                                                                                  num_steps,
                                                                                  loss.value(),
                                                                                  optimizer.optimizer.learning_rate,
                                                                                  time() - time_s))

        if num_steps % (iter_per_epoch // 2) == 0:
            CalculateBleu(model, test_data, 'val/main/bleu', device=-1, batch=args.batchsize // 4)()

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
            test_losses = []
            while True:
                test_batch = test_iter.next()
                in_arrays = seq2seq_pad_concat_convert(test_batch, -1)

                # Forward the test data
                model.set_dropout(0.0)
                loss_test = model(*in_arrays)

                # Calculate the accuracy
                test_losses.append(loss_test.value())

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break

            print('val_loss:{:.04f}'.format(np.mean(test_losses)))

            # CalculateBleu(model, test_data, 'val/main/bleu', device=args.gpu, batch=args.batchsize)()
            ############################################################

    # If you want to change a logging interval, change this number
    log_trigger = (min(200, iter_per_epoch // 2), 'iteration')

    def floor_step(trigger):
        floored = trigger[0] - trigger[0] % log_trigger[0]
        if floored <= 0:
            floored = trigger[0]
        return (floored, trigger[1])

    # Validation every half epoch
    eval_trigger = floor_step((iter_per_epoch // 2, 'iteration'))
    record_trigger = training.triggers.MinValueTrigger(
        'val/main/perp', eval_trigger)

    evaluator = extensions.Evaluator(test_iter, model, converter=seq2seq_pad_concat_convert, device=args.gpu0)
    evaluator.default_name = 'val'
    trainer.extend(evaluator, trigger=eval_trigger)

    # Use Vaswan's magical rule of learning rate(Eq. 3 in the paper)
    # But, the hyperparamter in the paper seems to work well
    # only with a large batchsize.
    # If you run on popular setup (e.g. size=48 on 1 GPU),
    # you may have to change the hyperparamter.
    # I scaled learning rate by 0.5 consistently.
    # ("scale" is always multiplied to learning rate.)

    # If you use a shallow layer network (<=2),
    # you may not have to change it from the paper setting.
    if not args.use_fixed_lr:
        trainer.extend(
            VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=32000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=4000, scale=0.5),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=16000, scale=1.),
            # VaswaniRule('alpha', d=args.unit, warmup_steps=64000, scale=1.),
            trigger=(1, 'iteration'))

    observe_alpha = extensions.observe_value('alpha', lambda trainer: trainer.updater.get_optimizer('main').alpha)
    trainer.extend(observe_alpha, trigger=(1, 'iteration'))

    # CalculateBleu(model, test_data, 'val/main/bleu', hyp_dev_path, target_vocab,
    #               device=args.gpu0, batch=args.batchsize // 4)(trainer)

    # Only if a model gets best validation score,
    # save (overwrite) the model
    trainer.extend(extensions.snapshot_object(model, 'best_model.npz'), trigger=record_trigger)

    def translate_one(source, target):
        words = preprocess.split_sentence(source)
        print('# source : ' + ' '.join(words))
        x = model.xp.array(
            [source_ids.get(w, 1) for w in words], 'i')
        # ys = model.translate([x], beam=5)[0]
        ys = model.translate([x], beam=1)[0]
        words = [target_words[y] for y in ys]
        print('#  result : ' + ' '.join(words))
        print('#  expect : ' + target)

    @chainer.training.make_extension(trigger=(200, 'iteration'))
    def translate(trainer):
        translate_one(
            'Who are we ?',
            'Qui sommes-nous?')
        translate_one(
            'And it often costs over a hundred dollars ' +
            'to obtain the required identity card .',
            'Or, il en coûte souvent plus de cent dollars ' +
            'pour obtenir la carte d\'identité requise.')

        source, target = test_data[numpy.random.choice(len(test_data))]
        source = ' '.join([source_words[i] for i in source])
        target = ' '.join([target_words[i] for i in target])
        translate_one(source, target)

    # Gereneration Test
    # trainer.extend(
    #     translate,
    #     trigger=(min(200, iter_per_epoch), 'iteration'))

    if not args.no_bleu:
        trainer.extend(
            CalculateBleu(model, test_data, 'val/main/bleu', hyp_dev_path, target_vocab, ref_dev_path,
                          device=args.gpu0, batch=args.batchsize // 4),
            trigger=floor_step((iter_per_epoch, 'iteration')))

    # Log
    trainer.extend(extensions.LogReport(trigger=log_trigger),
                   trigger=log_trigger)

    trainer.extend(extensions.PrintReport(['epoch', 'iteration',
                                           'main/loss', 'val/main/loss',
                                           'main/perp', 'val/main/perp',
                                           'main/acc', 'val/main/acc',
                                           'val/main/bleu',
                                           'alpha',
                                           'elapsed_time']),
                   trigger=log_trigger)

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()
