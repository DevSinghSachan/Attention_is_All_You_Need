# encoding: utf-8

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import utils

cudnn.benchmark = True


class LayerNorm(nn.Module):
    """Layer normalization module.
    Code adapted from OpenNMT-py open-source toolkit on 08/01/2018:
    URL: https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/modules/UtilClass.py#L24"""
    def __init__(self, d_hid, eps=1e-3):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.a_2 = nn.Parameter(torch.ones(d_hid), requires_grad=True)
        self.b_2 = nn.Parameter(torch.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(1) == 1:
            return z
        mu = torch.mean(z, dim=1)
        sigma = torch.std(z, dim=1)
        # HACK. PyTorch is changing behavior
        if mu.dim() == 1:
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
        ln_out = (z - mu.expand_as(z)) / (sigma.expand_as(z) + self.eps)
        ln_out = ln_out.mul(self.a_2.expand_as(ln_out)) + self.b_2.expand_as(ln_out)
        return ln_out


def sentence_block_embed(embed, x):
    """Computes sentence-level embedding representation from word-ids.

    :param embed: nn.Embedding() Module
    :param x: Tensor of batched word-ids
    :return: Tensor of shape (batchsize, dimension, sentence_length)
    """
    batch, length = x.shape
    _, units = embed.weight.size()
    e = embed(x).transpose(1, 2).contiguous()
    assert (e.size() == (batch, units, length))
    return e


def seq_func(func, x, reconstruct_shape=True):
    """Change implicitly function's input x from ndim=3 to ndim=2

    :param func: function to be applied to input x
    :param x: Tensor of batched sentence level word features
    :param reconstruct_shape: boolean, if the output needs to be of the same shape as input x
    :return: Tensor of shape (batchsize, dimension, sentence_length) or (batchsize x sentence_length, dimension)
    """
    batch, units, length = x.shape
    e = torch.transpose(x, 1, 2).contiguous().view(batch * length, units)
    e = func(e)
    if not reconstruct_shape:
        return e
    e = torch.transpose(e.view((batch, length, units)), 1, 2).contiguous()
    assert (e.shape == (batch, units, length))
    return e


class LayerNormSent(LayerNorm):
    """Position-wise layer-normalization layer for array of shape (batchsize, dimension, sentence_length)."""
    def __init__(self, n_units, eps=1e-3):
        super(LayerNormSent, self).__init__(n_units, eps=eps)

    def forward(self, x):
        y = seq_func(super(LayerNormSent, self).forward, x)
        return y


class LinearSent(nn.Module):
    """Position-wise Linear Layer for sentence block. array of shape (batchsize, dimension, sentence_length)."""
    def __init__(self, input_dim, output_dim, bias=True):
        super(LinearSent, self).__init__()
        self.L = nn.Linear(input_dim, output_dim, bias=bias)
        self.L.weight.data.uniform_(-3./input_dim, 3./input_dim)
        if bias:
            self.L.bias.data.fill_(0.)
        self.output_dim = output_dim

    def forward(self, x):
        output = self.L.weight.matmul(x)
        if self.L.bias is not None:
            output += self.L.bias.unsqueeze(-1)
        return output


class MultiHeadAttention(nn.Module):
    """Multi Head Attention Layer for Sentence Blocks. For computational efficiency, dot-product to calculate
    query-key scores is performed in all the heads together."""
    def __init__(self, n_units, h=8, attn_dropout=False, dropout=0.2):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = LinearSent(n_units, n_units, bias=False)
        self.W_K = LinearSent(n_units, n_units, bias=False)
        self.W_V = LinearSent(n_units, n_units, bias=False)
        self.finishing_linear_layer = LinearSent(n_units, n_units, bias=False)
        self.h = h
        self.scale_score = 1. / (n_units // h) ** 0.5
        self.attn_dropout = attn_dropout
        if attn_dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, z=None, mask=None):
        h = self.h
        Q = self.W_Q(x)
        if z is None:
            K, V = self.W_K(x), self.W_V(x)
        else:
            K, V = self.W_K(z), self.W_V(z)

        batch, n_units, n_querys = Q.shape
        _, _, n_keys = K.shape

        # Calculate attention scores with mask for zero-padded areas
        # Perform multi-head attention using pseudo batching all together at once for efficiency
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
        batch_A = batch_A.masked_fill(batch_A != batch_A, 0.)
        assert (batch_A.shape == (batch * h, n_querys, n_keys))

        # Attention Dropout
        if self.attn_dropout:
            batch_A = self.dropout(batch_A)

        # Calculate Weighted Sum
        V = V.transpose(1, 2).contiguous()
        C = torch.transpose(torch.bmm(batch_A, V), 1, 2).contiguous()
        assert (C.shape == (batch * h, n_units // h, n_querys))

        # Joining the Multiple Heads
        C = torch.cat(torch.chunk(C, h, dim=0), dim=1)
        assert (C.shape == (batch, n_units, n_querys))

        # Final linear layer
        C = self.finishing_linear_layer(C)
        return C


class FeedForwardLayer(nn.Module):
    def __init__(self, n_units):
        super(FeedForwardLayer, self).__init__()
        n_inner_units = n_units * 4
        self.W_1 = LinearSent(n_units, n_inner_units)
        self.act = nn.ReLU()
        self.W_2 = LinearSent(n_inner_units, n_units)

    def forward(self, e):
        e = self.W_1(e)
        e = self.act(e)
        e = self.W_2(e)
        return e


class EncoderLayer(nn.Module):
    def __init__(self, n_units, h=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(EncoderLayer, self).__init__()
        self.layer_norm = layer_norm
        self.self_attention = MultiHeadAttention(n_units, h, attn_dropout, dropout)
        self.dropout1 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_1 = LayerNormSent(n_units, eps=1e-3)
        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout2 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_2 = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, xx_mask):
        sub = self.self_attention(e, mask=xx_mask)
        e = e + self.dropout1(sub)
        if self.layer_norm:
            e = self.ln_1(e)

        sub = self.feed_forward(e)
        e = e + self.dropout2(sub)
        if self.layer_norm:
            e = self.ln_2(e)
        return e


class DecoderLayer(nn.Module):
    def __init__(self, n_units, h=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(DecoderLayer, self).__init__()
        self.layer_norm = layer_norm
        self.self_attention = MultiHeadAttention(n_units, h, attn_dropout, dropout)
        self.dropout1 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_1 = LayerNormSent(n_units, eps=1e-3)

        self.source_attention = MultiHeadAttention(n_units, h, attn_dropout, dropout)
        self.dropout2 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_2 = LayerNormSent(n_units, eps=1e-3)

        self.feed_forward = FeedForwardLayer(n_units)
        self.dropout3 = nn.Dropout(dropout)
        if layer_norm:
            self.ln_3 = LayerNormSent(n_units, eps=1e-3)

    def forward(self, e, s, xy_mask, yy_mask):
        sub = self.self_attention(e, mask=yy_mask)
        e = e + self.dropout1(sub)
        if self.layer_norm:
            e = self.ln_1(e)

        sub = self.source_attention(e, s, mask=xy_mask)
        e = e + self.dropout2(sub)
        if self.layer_norm:
            e = self.ln_2(e)

        sub = self.feed_forward(e)
        e = e + self.dropout3(sub)
        if self.layer_norm:
            e = self.ln_3(e)
        return e


class Encoder(nn.Module):
    def __init__(self, n_layers, n_units, h=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(Encoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            layer = EncoderLayer(n_units, h, dropout, layer_norm, attn_dropout)
            self.layers.append(layer)

    def forward(self, e, xx_mask):
        for layer in self.layers:
            e = layer(e, xx_mask)
        return e


class Decoder(nn.Module):
    def __init__(self, n_layers, n_units, h=8, dropout=0.2, layer_norm=True, attn_dropout=False):
        super(Decoder, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(1, n_layers + 1):
            layer = DecoderLayer(n_units, h, dropout, layer_norm, attn_dropout)
            self.layers.append(layer)

    def forward(self, e, source, xy_mask, yy_mask):
        for layer in self.layers:
            e = layer(e, source, xy_mask, yy_mask)
        return e


class Transformer(nn.Module):
    def __init__(self, n_layers, n_source_vocab, n_target_vocab, n_units, h=8, dropout=0.1, max_length=500,
                 label_smoothing=False, embed_position=False, layer_norm=True, tied=True, attn_dropout=False):
        super(Transformer, self).__init__()
        self.embed_x = nn.Embedding(n_source_vocab, n_units, padding_idx=0)
        self.embed_y = nn.Embedding(n_target_vocab, n_units, padding_idx=0)
        self.embed_x.weight.data.uniform_(-3. / n_source_vocab, 3. / n_source_vocab)
        self.embed_y.weight.data.uniform_(-3. / n_target_vocab, 3. / n_target_vocab)

        self.embed_dropout = nn.Dropout(dropout)
        self.encoder = Encoder(n_layers, n_units, h, dropout, layer_norm, attn_dropout)
        self.decoder = Decoder(n_layers, n_units, h, dropout, layer_norm, attn_dropout)

        if embed_position:
            self.embed_pos = nn.Embedding(max_length, n_units, padding_idx=0)

        self.affine = nn.Linear(n_units, n_target_vocab, bias=False)
        if tied:
            self.affine.weight = self.embed_y.weight

        self.n_layers = n_layers
        self.n_units = n_units
        self.n_target_vocab = n_target_vocab
        self.dropout = dropout
        self.label_smoothing = label_smoothing
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
        # return F.linear(h, self.embed_y.weight)
        return self.affine(h)

    def output_and_loss(self, h_block, t_block):
        batch, units, length = h_block.shape

        # Output (all together at once for efficiency)
        concat_logit_block = seq_func(self.affine, h_block, reconstruct_shape=False)
        rebatch, _ = concat_logit_block.shape

        # Make target
        concat_t_block = t_block.view(rebatch)
        ignore_mask = (concat_t_block >= 1).float()
        n_token = torch.sum(ignore_mask)
        normalizer = n_token

        if self.label_smoothing:
            log_prob = F.log_softmax(concat_logit_block, dim=1)
            broad_ignore_mask = ignore_mask[:, None].expand_as(concat_logit_block)
            pre_loss = ignore_mask * log_prob[np.arange(rebatch), concat_t_block]
            loss = -1. * torch.sum(pre_loss) / normalizer
        else:
            loss = F.cross_entropy(concat_logit_block, concat_t_block, ignore_index=0)

        accuracy = utils.accuracy(concat_logit_block, concat_t_block, ignore_index=0)
        perplexity = torch.exp(loss.data)

        if self.label_smoothing:
            pre_loss = (1 - self.label_smoothing) * loss
            ls_loss = -1. / self.n_target_vocab * broad_ignore_mask * log_prob
            ls_loss = torch.sum(ls_loss) / normalizer
            loss = pre_loss + (self.label_smoothing * ls_loss)

        return loss, accuracy, perplexity

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
        x_block = utils.source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        y_block = np.full((batch, 1), 3, dtype=x_block.dtype)  # bos
        eos_flags = np.zeros((batch,), dtype=x_block.dtype)

        x_block, y_block = Variable(torch.LongTensor(x_block)), Variable(torch.LongTensor(y_block))
        if torch.cuda.is_available():
            x_block, y_block = x_block.cuda(), y_block.cuda()

        result = []
        for i in range(max_length):
            log_prob_tail = self(x_block, y_block, y_out_block=None, get_prediction=True)
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
        x_block = utils.source_pad_concat_convert(x_block, device=None)
        batch, x_length = x_block.shape
        assert batch == 1, 'Batch processing is not supported now.'

        y_block = np.full((batch, 1), 3, dtype=x_block.dtype)  # bos
        eos_flags = np.zeros((batch * beam,), dtype=x_block.dtype)
        # eos_flags = torch.zeros((batch * beam)).type(torch.IntTensor)
        beam_scores = torch.zeros(1).type(utils.FLOAT_TYPE)
        result = [[3]] * batch * beam

        x_block, y_block = Variable(torch.LongTensor(x_block)), Variable(torch.LongTensor(y_block))
        if torch.cuda.is_available():
            x_block, y_block = x_block.cuda(), y_block.cuda()

        for i in range(max_length):
            log_prob_tail = self(x_block, y_block, y_out_block=None, get_prediction=True)
            log_prob_tail = F.log_softmax(log_prob_tail, dim=1)

            # Get the top-k scores and word-ids
            scores_array, ids_array = torch.topk(log_prob_tail.data, k=beam, dim=1)

            # Compute the cumulative running sum of beam scores
            if beam_scores.shape[0] != scores_array.shape[0]:
                beam_scores = beam_scores.expand(scores_array.shape[0], beam_scores.shape[1])

            cumulative_scores = beam_scores + scores_array

            # Reshaping current cumulative scores in 1-column
            scores_col = cumulative_scores.view(cumulative_scores.nelement())
            ids_col = ids_array.view(ids_array.nelement())

            # Get top-k from total candidates at every step
            top_scores, top_idxs = torch.topk(scores_col, beam, dim=0)
            new_ids = ids_col[top_idxs]
            new_beam_index = top_idxs / beam

            # Updating the beam_score from the top score
            new_result = [[]] * batch * beam
            new_beam_scores = torch.zeros(beam).type(utils.FLOAT_TYPE)
            for j in range(beam):
                k = new_beam_index[j]
                new_beam_scores[j] = beam_scores[k] + top_scores[j]
                new_result[j] = result[k] + [new_ids[j]]

            result = deepcopy(new_result)
            beam_scores = deepcopy(new_beam_scores)

            y_block = Variable(torch.LongTensor(result))

            if x_block.shape[0] != y_block.shape[0]:
                x_block = x_block.expand(y_block.shape[0], x_block.shape[1])

            if torch.cuda.is_available():
                y_block = y_block.cuda()

            eos_flags += (new_ids == 1)
            if np.all(eos_flags):
            # if torch.nonzero(eos_flags):
                break

        outs = [[wi for wi in sent if wi not in [3, 1]] for sent in result]
        outs = [sent if sent else [1] for sent in outs]
        return outs[0]
