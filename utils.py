import numpy as np
import torch
from torch.autograd import Variable
from chainer.dataset import convert

if torch.cuda.is_available():
    FLOAT_TYPE = torch.cuda.FloatTensor
    INT_TYPE = torch.cuda.IntTensor
    LONG_TYPE = torch.cuda.LongTensor
    BYTE_TYPE = torch.cuda.ByteTensor
else:
    FLOAT_TYPE = torch.FloatTensor
    INT_TYPE = torch.IntTensor
    LONG_TYPE = torch.LongTensor
    BYTE_TYPE = torch.ByteTensor


class Accuracy(object):
    def __init__(self, ignore_index=None):
        self.ignore_index = ignore_index

    def __call__(self, y, t):
        if self.ignore_index is not None:
            mask = (t == self.ignore_index)
            ignore_cnt = torch.sum(mask.float())
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            pred = pred.masked_fill(mask, self.ignore_index)
            count = torch.sum((pred == t).float()) - ignore_cnt
            total = torch.numel(t) - ignore_cnt

            if total == 0:
                return torch.FloatTensor([0.0])
            else:
                return count / total
        else:
            _, pred = torch.max(y, dim=1)
            pred = pred.view(t.shape)
            return torch.mean((pred == t).float())


def accuracy(y, t, ignore_index=None):
    return Accuracy(ignore_index=ignore_index)(y, t)


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
    x_block, y_in_block, y_out_block = Variable(torch.LongTensor(x_block).type(LONG_TYPE)), \
                                       Variable(torch.LongTensor(y_in_block).type(LONG_TYPE)), \
                                       Variable(torch.LongTensor(y_out_block).type(LONG_TYPE))

    return x_block, y_in_block, y_out_block


def source_pad_concat_convert(x_seqs, device, eos_id=1, bos_id=3):
    x_block = convert.concat_examples(x_seqs, device, padding=0)

    # add eos
    x_block = np.pad(x_block, ((0, 0), (0, 1)), 'constant', constant_values=0)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = eos_id

    x_block = np.pad(x_block, ((0, 0), (1, 0)), 'constant', constant_values=bos_id)
    return x_block

