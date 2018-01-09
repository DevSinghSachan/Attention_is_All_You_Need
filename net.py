# encoding: utf-8

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from train import source_pad_concat_convert


def sentence_block_embed(embed, x):
    batch, length = x.shape
    _, units = embed.weight.size()
    e = embed(x).transpose(1, 2).contiguous()
    assert (e.size() == (batch, units, length))
    return e


def seq_func(func, x, reconstruct_shape=True):
    """ Change implicitly function's target to ndim=3
    Apply a given function for array of ndim 3,
    shape (batchsize, dimension, sentence_length),
    instead for array of ndim 2.
    """
    batch, units, length = x.shape
    e = torch.transpose(x, 1, 2).contiguous().view(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    out_units = e.shape[1]

    e = torch.transpose(e.view((batch, length, out_units)), 1, 2).contiguous()
    assert (e.shape == (batch, out_units, length))
    return e


class LayerNormalizationSentence(torch.nn.Module):
    """ Position-wise Linear Layer for Sentence Block

    Position-wise layer-normalization layer for array of shape
    (batchsize, dimension, sentence_length).

    """

    def __init__(self, *args, **kwargs):
        super(LayerNormalizationSentence, self).__init__()

    def forward(self, x):
        # y = seq_func(super(LayerNormalizationSentence, self).forward, x)
        # return y
        return x


class LinearSent(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        self.L.weight.data.uniform_(-3. / input_dim, 3. / input_dim)
        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, input_expr):
        batch_size, _, seq_len = input_expr.shape

        # output = seq_func(self.L, input_expr)
        output = self.L.weight.matmul(input_expr)
        if self.L.bias is not None:
            output += self.L.bias.unsqueeze(-1)
        return output


# differentiable equivalent of np.where
# cond could be a FloatTensor with zeros and ones
def where(cond, x_1, x_2):
    cond = cond.float()
    return (cond * x_1) + ((1-cond) * x_2)


class MultiHeadAttention(torch.nn.Module):
    """ Multi Head Attention Layer for Sentence Blocks
    For batch computation efficiency, dot product to calculate query-key
    scores is performed all heads together.
    """

    def __init__(self, n_units, h=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = LinearSent(n_units, n_units, bias=False)
        self.W_K = LinearSent(n_units, n_units, bias=False)
        self.W_V = LinearSent(n_units, n_units, bias=False)
        self.finishing_linear_layer = LinearSent(n_units, n_units, bias=False)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.dropout = dropout

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)

        if z is None:
            K, V = self.W_K(x), self.W_V(x)
        else:
            K, V = self.W_K(z), self.W_V(z)

        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate Attention Scores with Mask for Zero-padded Areas
        # Perform Multi-head Attention using pseudo batching
        # all together at once for efficiency
        Q = torch.cat(torch.chunk(Q, h, dim=1), dim=0)
        K = torch.cat(torch.chunk(K, h, dim=1), dim=0)
        V = torch.cat(torch.chunk(V, h, dim=1), dim=0)

        assert (Q.shape == (batch * h, n_units // h, n_querys))
        assert (K.shape == (batch * h, n_units // h, n_keys))
        assert (V.shape == (batch * h, n_units // h, n_keys))

        mask = torch.cat([mask] * h, dim=0)
        Q = Q.transpose(1, 2).contiguous()
        batch_A = torch.bmm(Q, K) * self.scale_score

        batch_A = batch_A.masked_fill(1. - mask, -np.inf)
        batch_A = F.softmax(batch_A, dim=2)

        # Replaces 'NaN' with zeros and other values with the original ones
        # Currently torch.where is only supported in CPU
        # batch_A = torch.where((batch_A != batch_A).cpu(), Variable(torch.zeros(batch_A.shape)), batch_A.cpu())

        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)

        # if torch.cuda.is_available():
        #     batch_A = batch_A.cuda()
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # Calculate Weighted Sum
        V = V.transpose(1, 2).contiguous()
        C = torch.transpose(torch.bmm(batch_A, V), 1, 2).contiguous()
        assert (C.shape == (batch * h, n_units // h, n_querys))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=1)
        assert (C.shape == (batch, n_units, n_querys))
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(torch.nn.Module):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        self.W_1 = LinearSent(n_units, n_inner_units)
        self.W_2 = LinearSent(n_inner_units, n_units)
        self.act = nn.ReLU()

    def forward(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(torch.nn.Module):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_units, h)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)

        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)

    def forward(self, e, xx_mask):
        sub = self.self_attention(e, mask=xx_mask)
        e = e + self.dropout1(sub)
        e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + self.dropout2(sub)
        e = self.ln_2(e)
        return e


class DecoderLayer(torch.nn.Module):
    def __init__(self, n_units, h=8, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(n_units, h)
        self.dropout1 = nn.Dropout(dropout)
        self.ln_1 = LayerNormalizationSentence(n_units, eps=1e-6)

        self.source_attention = MultiHeadAttention(n_units, h)
        self.dropout2 = nn.Dropout(dropout)
        self.ln_2 = LayerNormalizationSentence(n_units, eps=1e-6)

        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout3 = nn.Dropout(dropout)
        self.ln_3 = LayerNormalizationSentence(n_units, eps=1e-6)

    def forward(self, e, s, xy_mask, yy_mask):
        sub = self.self_attention(e, mask=yy_mask)
        e = e + self.dropout1(sub)
        e = self.ln_1(e)

        sub = self.source_attention(e, s, mask=xy_mask)
        e = e + self.dropout2(sub)
        e = self.ln_2(e)

        sub = self.feed_forward(e)
        e = e + self.dropout3(sub)
        e = self.ln_3(e)
        return e


class Encoder(torch.nn.Module):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            layer = EncoderLayer(n_units, h, dropout)
            self.layers.append(layer)

    def forward(self, e, xx_mask):
        for layer in self.layers:
            e = layer(e, xx_mask)
        return e


class Decoder(torch.nn.Module):
    def __init__(self, n_layers, n_units, h=8, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            layer = DecoderLayer(n_units, h, dropout)
            self.layers.append(layer)

    def forward(self, e, source, xy_mask, yy_mask):
        for layer in self.layers:
            e = layer(e, source, xy_mask, yy_mask)
        return e


class Transformer(torch.nn.Module):
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, h=8, dropout=0.1, max_length=500,
                 use_label_smoothing=False, embed_position=False):
        super(Transformer, self).__init__()
        self.embed_x = nn.Embedding(n_source_vocab, n_units, padding_idx=0)
        self.embed_y = nn.Embedding(n_target_vocab, n_units, padding_idx=0)

        self.embed_x.weight.data.uniform_(-3. / n_source_vocab, 3. / n_source_vocab)
        self.embed_y.weight.data.uniform_(-3. / n_target_vocab, 3. / n_target_vocab)

        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, n_units, h, dropout)
        self.decoder = Decoder(n_layers, n_units, h, dropout)
        if embed_position:
            self.embed_pos = nn.Embedding(max_length, n_units, padding_idx=0)

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.affine = nn.Linear(n_units, n_target_vocab, bias=False)
        self.dropout = dropout
        self.use_label_smoothing = use_label_smoothing
        self.initialize_position_encoding(max_length, n_units)
        self.scale_emb = self.n_units ** 0.5

    def initialize_position_encoding(self, length, emb_dim):
        channels = emb_dim
        position = np.arange(length, dtype='f')
        num_timescales = channels // 2
        log_timescale_increment = (np.log(10000. / 1.) / (float(num_timescales) - 1))
        inv_timescales = 1. * np.exp(np.arange(num_timescales).astype('f') * -log_timescale_increment)
        scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)
        signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
        signal = np.reshape(signal, [1, length, channels])
        position_encoding_block = np.transpose(signal, (0, 2, 1))

        self.position_encoding_block = nn.Parameter(torch.FloatTensor(position_encoding_block), requires_grad=False)
        self.register_parameter("Position Encoding Block", self.position_encoding_block)

    # def to_gpu(self, device=None):
    #     # super(chainer.Chain, self).to_gpu(device)
    #     # xp = cuda.get_array_module()
    #     #import cupy
    #     with cuda._get_device(device):
    #         super(chainer.Chain, self).to_gpu()
    #         d = self.__dict__
    #         for name in self._children:
    #             d[name].to_gpu()
    #         self.position_encoding_block = cupy.asarray(self.position_encoding_block)

    def make_input_embedding(self, embed, block):
        batch, length = block.shape
        emb_block = sentence_block_embed(embed, block) * self.scale_emb
        emb_block += self.position_encoding_block[:, :, :length]

        if hasattr(self, 'embed_pos'):
            emb_block += sentence_block_embed(self.embed_pos,
                                              self.xp.broadcast_to(self.xp.arange(length).astype('i')[None, :],
                                                                   block.shape))
        emb_block = self.embed_dropout(emb_block)
        return emb_block

    def make_attention_mask(self, source_block, target_block):
        mask = (target_block[:, None, :] >= 1) * \
               (source_block[:, :, None] >= 1)
        # (batch, source_length, target_length)
        return mask

    def make_history_mask(self, block):
        batch, length = block.shape
        arange = np.arange(length)
        history_mask = (arange[None,] <= arange[:, None])[None,]
        history_mask = np.broadcast_to(history_mask, (batch, length, length))
        history_mask = history_mask.astype(np.int32)
        history_mask = Variable(torch.ByteTensor(history_mask))
        if torch.cuda.is_available():
            history_mask = history_mask.cuda()
        return history_mask

    def output(self, h):
        return F.linear(h, self.embed_y.weight)

    def output_and_loss(self, h_block, t_block):
        batch, units, length = h_block.shape
        # Output (all together at once for efficiency)
        # concat_logit_block = seq_func(self.affine, h_block, reconstruct_shape=False)
        concat_logit_block = seq_func(self.output, h_block, reconstruct_shape=False)
        rebatch, _ = concat_logit_block.shape

        # Make target
        concat_t_block = t_block.view(rebatch)
        ignore_mask = (concat_t_block >= 1)
        loss = F.cross_entropy(concat_logit_block, concat_t_block, ignore_index=0)
        # accuracy = F.accuracy(concat_logit_block, concat_t_block, ignore_label=0)
        return loss

    def forward(self, x_block, y_in_block, y_out_block, get_prediction=False):
        batch, x_length = x_block.shape
        batch, y_length = y_in_block.shape

        # Make Embedding
        ex_block = self.make_input_embedding(self.embed_x, x_block)
        ey_block = self.make_input_embedding(self.embed_y, y_in_block)

        # Make Masks
        xx_mask = self.make_attention_mask(x_block, x_block)
        xy_mask = self.make_attention_mask(y_in_block, x_block)
        yy_mask = self.make_attention_mask(y_in_block, y_in_block)
        yy_mask *= self.make_history_mask(y_in_block)

        # Encode Sources
        z_blocks = self.encoder(ex_block, xx_mask)
        # [(batch, n_units, x_length), ...]

        # Encode Targets with Sources (Decode without Output)
        h_block = self.decoder(ey_block, z_blocks, xy_mask, yy_mask)
        # (batch, n_units, y_length)

        if get_prediction:
            return self.output(h_block[:, :, -1])
        else:
            return self.output_and_loss(h_block, y_out_block)

    def translate(self, x_block, max_length=50, beam=5):
        if beam:
            return self.translate_beam(x_block, max_length, beam)

        # TODO: efficient inference by re-using result
        x_block = source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        y_block = np.full((batch, 1), 3, dtype=x_block.dtype)  # bos
        eos_flags = np.zeros((batch,), dtype=x_block.dtype)

        x_block, y_block = Variable(torch.LongTensor(x_block)), Variable(torch.LongTensor(y_block))
        if torch.cuda.is_available():
            x_block, y_block = x_block.cuda(), y_block.cuda()

        result = []
        for i in range(max_length):
            log_prob_tail = self(x_block, y_block, y_block, get_prediction=True)
            # ys = np.argmax(log_prob_tail.data.cpu().numpy(), axis=1).astype('i')
            _, ys = torch.max(log_prob_tail, dim=1)
            y_block = torch.cat([y_block.detach(), ys[:, None]], dim=1)
            ys = ys.data.cpu().numpy()
            result.append(ys)
            eos_flags += (ys == 1)
            if np.all(eos_flags):
                break

        result = np.stack(result).T

        # Remove EOS tags
        outs = []
        for y in result:
            inds = np.argwhere(y==1)
            if len(inds) > 0:
                y = y[:inds[0, 0]]
            if len(y) == 0:
                y = np.array([1], 'i')
            outs.append(y)
        return outs

    def translate_beam(self, x_block, max_length=50, beam=5):
        # TODO: efficient inference by re-using result
        # TODO: batch processing
        with chainer.no_backprop_mode():
            with chainer.using_config('train', False):
                x_block = source_pad_concat_convert(
                    x_block, device=None)
                batch, x_length = x_block.shape
                assert batch == 1, 'Batch processing is not supported now.'
                y_block = self.xp.full(
                    (batch, 1), 2, dtype=x_block.dtype)  # bos
                eos_flags = self.xp.zeros(
                    (batch * beam,), dtype=x_block.dtype)
                sum_scores = self.xp.zeros(1, 'f')
                result = [[2]] * batch * beam
                for i in range(max_length):
                    log_prob_tail = self(x_block, y_block, y_block,
                                         get_prediction=True)

                    ys_list, ws_list = get_topk(
                        log_prob_tail.data, beam, axis=1)
                    ys_concat = self.xp.concatenate(ys_list, axis=0)
                    sum_ws_list = [ws + sum_scores for ws in ws_list]
                    sum_ws_concat = self.xp.concatenate(sum_ws_list, axis=0)

                    # Get top-k from total candidates
                    idx_list, sum_w_list = get_topk(sum_ws_concat, beam, axis=0)
                    # idx_concat = self.xp.stack(idx_list, axis=0)
                    idx_concat = self.xp.concatenate([self.xp.expand_dims(x, axis=0) for x in idx_list], axis=0)
                    ys = ys_concat[idx_concat]
                    # sum_scores = self.xp.stack(sum_w_list, axis=0)
                    sum_scores = self.xp.concatenate([self.xp.expand_dims(x, axis=0) for x in sum_w_list], axis=0)

                    if i != 0:
                        old_idx_list = (idx_concat % beam).tolist()
                    else:
                        old_idx_list = [0] * beam

                    result = [result[idx] + [y]
                              for idx, y in zip(old_idx_list, ys.tolist())]

                    y_block = self.xp.array(result).astype('i')
                    if x_block.shape[0] != y_block.shape[0]:
                        x_block = self.xp.broadcast_to(
                            x_block, (y_block.shape[0], x_block.shape[1]))
                    eos_flags += (ys == 0)
                    if self.xp.all(eos_flags):
                        break

        outs = [[wi for wi in sent if wi not in [2, 0]] for sent in result]
        outs = [sent if sent else [0] for sent in outs]
        return outs


def get_topk(x, k=5, axis=1):
    ids_list = []
    scores_list = []
    xp = cuda.get_array_module(x)
    for i in range(k):
        ids = xp.argmax(x, axis=axis).astype('i')
        if axis == 0:
            scores = x[ids]
            x[ids] = - float('inf')
        else:
            scores = x[xp.arange(ids.shape[0]), ids]
            x[xp.arange(ids.shape[0]), ids] = - float('inf')
        ids_list.append(ids)
        scores_list.append(scores)
    return ids_list, scores_list
