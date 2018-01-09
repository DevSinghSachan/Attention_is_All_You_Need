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
from subfuncs import VaswaniRule, TransformerAdamTrainer
import general_utils
from util import get_args

if torch.cuda.is_available():
    torch.cuda.set_device(0)


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
    config = get_args()

    print(json.dumps(config.__dict__, indent=4))

    # Check file
    source_path = os.path.join(config.input, config.source)
    source_vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + preprocess.count_words(source_path, config.source_vocab)
    source_data = preprocess.make_dataset(source_path, source_vocab)

    target_path = os.path.join(config.input, config.target)
    target_vocab = ['<pad>', '<eos>', '<unk>', '<bos>'] + preprocess.count_words(target_path, config.target_vocab)
    target_data = preprocess.make_dataset(target_path, target_vocab)
    assert len(source_data) == len(target_data)

    print("Source Vocab: {}".format(len(source_vocab)))
    print("Target Vocab: {}".format(len(target_vocab)))

    print('Original training data size: %d' % len(source_data))
    train_data = [(s, t) for s, t in six.moves.zip(source_data, target_data) if 0 < len(s) < 50 and 0 < len(t) < 50]
    print('Filtered training data size: %d' % len(train_data))

    source_path = os.path.join(config.input, config.source_valid)
    source_data = preprocess.make_dataset(source_path, source_vocab)
    target_path = os.path.join(config.input, config.target_valid)
    target_data = preprocess.make_dataset(target_path, target_vocab)
    assert len(source_data) == len(target_data)
    test_data = [(s, t) for s, t in six.moves.zip(source_data, target_data) if 0 < len(s) and 0 < len(t)]

    hyp_dev_path = os.path.join(config.input, config.hyp_dev)
    hyp_test_path = os.path.join(config.input, config.hyp_test)
    ref_dev_path = os.path.join(config.input, config.target_valid_raw)

    source_ids = {word: index for index, word in enumerate(source_vocab)}
    target_ids = {word: index for index, word in enumerate(target_vocab)}

    target_words = {i: w for w, i in target_ids.items()}
    source_words = {i: w for w, i in source_ids.items()}

    # Define Model
    model = net.Transformer(config.layer,
                            min(len(source_ids), len(source_words)),
                            min(len(target_ids), len(target_words)),
                            config.unit,
                            h=config.head,
                            dropout=config.dropout,
                            max_length=500,
                            use_label_smoothing=config.use_label_smoothing,
                            embed_position=config.embed_position)

    if config.gpu >= 0:
        model.cuda(config.gpu)

    # Setup Optimizer
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = TransformerAdamTrainer(model)
    train_iter = chainer.iterators.SerialIterator(train_data, config.batchsize)
    test_iter = chainer.iterators.SerialIterator(test_data, config.batchsize, repeat=False, shuffle=False)

    iter_per_epoch = len(train_data) // config.batchsize
    print('Number of iter/epoch =', iter_per_epoch)
    print("epoch \t steps \t train_loss \t lr \t time")
    prog = general_utils.Progbar(target=iter_per_epoch)
    num_steps = 0
    time_s = time()

    while train_iter.epoch < config.epoch:
        model.train()
        optimizer.zero_grad()
        num_steps += 1

        # ---------- One iteration of the training loop ----------
        train_batch = train_iter.next()
        in_arrays = seq2seq_pad_concat_convert(train_batch, -1)
        loss = model(*in_arrays)

        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm(model.parameters(), 5.0)
        optimizer.step()

        if config.debug:
            dummy_norm = 50.0
            norm = torch.nn.utils.clip_grad_norm(model.parameters(), dummy_norm)
            prog.update(num_steps, values=[("train loss", loss.data.cpu().numpy()[0]),], exact=[("norm", norm)])

        # if num_steps % (iter_per_epoch // 2) == 0:
        #     CalculateBleu(model, test_data, 'val/main/bleu', device=-1, batch=args.batchsize // 4)()

        # Check the validation accuracy of prediction after every epoch
        if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
            prog = general_utils.Progbar(target=iter_per_epoch)
            test_losses = []
            while True:
                model.eval()
                test_batch = test_iter.next()
                in_arrays = seq2seq_pad_concat_convert(test_batch, -1)

                # Forward the test data
                loss_test = model(*in_arrays)

                # Calculate the accuracy
                test_losses.append(loss_test.data.cpu().numpy())

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

    evaluator = extensions.Evaluator(test_iter, model, converter=seq2seq_pad_concat_convert, device=config.gpu0)
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
    if not config.use_fixed_lr:
        trainer.extend(
            VaswaniRule('alpha', d=config.unit, warmup_steps=4000, scale=1.),
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

    if not config.no_bleu:
        trainer.extend(
            CalculateBleu(model, test_data, 'val/main/bleu', hyp_dev_path, target_vocab, ref_dev_path,
                          device=config.gpu0, batch=config.batchsize // 4),
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
