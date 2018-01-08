from __future__ import division, generators

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import utils

"""This code was adapted from XNMT open-source toolkit on 01/10/2018. 
URL: https://github.com/neulab/xnmt/blob/master/xnmt/search_strategy.py"""


class PolynomialNormalization(object):
    """Dividing by the length (raised to some power (default 1))"""
    def __init__(self, m=1, apply_during_search=False):
        self.m = m
        self.apply_during_search = apply_during_search

    def lp(self, len):
        return pow(5 + len, self.m) / pow(5 + 1, self.m)

    def normalize_completed(self, completed_hyps, src_length=None):
        if not self.apply_during_search:
            for hyp in completed_hyps:
                hyp.score /= pow(len(hyp.id_list), self.m)

    def normalize_partial(self, score_so_far, score_to_add, new_len):
        if self.apply_during_search:
            return (score_so_far * self.lp(new_len - 1) + score_to_add) / self.lp(new_len)
        else:
            return score_so_far + score_to_add


class BeamSearch(object):
    def __init__(self, beam_size=5, max_len=50):
        self.beam_size = beam_size
        self.max_len = max_len
        self.len_norm = PolynomialNormalization(apply_during_search=True)
        self.entrs = []

    class Hypothesis:
        def __init__(self, score, id_list):
            self.score = score
            # self.state = state
            self.id_list = id_list

        def __str__(self):
            return "hypo S=%s ids=%s" % (self.score, self.id_list)

        def __repr__(self):
            return "hypo S=%s |ids|=%s" % (self.score, len(self.id_list))

    def generate_output(self, model, x_block):
        """
    :param decoder: decoder.Decoder subclass
    :param attender: attender.Attender subclass
    :param output_embedder: embedder.Embedder subclass
    :param dec_state: The decoder state
    :param src_length: length of src sequence, required for some types of length normalization
    :returns: (id list, score)
    """

        x_block = utils.source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        assert batch == 1, 'Batch processing is not supported now.'
        x_block = Variable(torch.LongTensor(x_block).type(utils.LONG_TYPE))

        active_hyp = [self.Hypothesis(0, [])]
        completed_hyp = []
        length = 0

        while len(completed_hyp) < self.beam_size and length < self.max_len:
            new_set = []
            for hyp in active_hyp:
                if length > 0:  # don't feed in the initial start-of-sentence token
                    if hyp.id_list[-1] == 1:
                        hyp.id_list = hyp.id_list[:-1]
                        completed_hyp.append(hyp)
                        continue

                y_block = Variable(torch.LongTensor([[3] + hyp.id_list])).type(utils.LONG_TYPE)

                log_prob_tail = model(x_block, y_block, y_out_block=None, get_prediction=True)
                score = F.log_softmax(log_prob_tail, dim=1).data.cpu().numpy()[0]
                top_ids = np.argpartition(score, max(-len(score), -self.beam_size))[-self.beam_size:]

                for cur_id in top_ids.tolist():
                    new_list = list(hyp.id_list)
                    new_list.append(cur_id)
                    log_score = self.len_norm.normalize_partial(hyp.score, score[cur_id], len(new_list))
                    new_set.append(self.Hypothesis(log_score, new_list))
            length += 1

            active_hyp = sorted(new_set, key=lambda x: x.score, reverse=True)[:self.beam_size]

        if len(completed_hyp) == 0:
            completed_hyp = active_hyp

        self.len_norm.normalize_completed(completed_hyp, x_length)

        result = sorted(completed_hyp, key=lambda x: x.score, reverse=True)[0]
        return result.id_list, result.score