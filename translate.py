# encoding: utf-8
from __future__ import unicode_literals, print_function

import json
import os
import io
import torch
from tqdm import tqdm

import preprocess
import utils
from config import get_translate_args


def save_output(hypotheses, vocab, outf):
    # Save the Hypothesis to output file
    with io.open(outf, 'w') as fp:
        for sent in hypotheses:
            words = [vocab[y] for y in sent]
            fp.write(' '.join(words) + '\n')


class TranslateText(object):
    def __init__(self, model, test_data, batch=50, max_length=50, beam_size=1):
        self.model = model
        self.test_data = test_data
        self.batch = batch
        self.device = -1
        self.max_length = max_length
        self.beam_size = beam_size

    def __call__(self):
        self.model.eval()
        hypotheses = []
        for i in tqdm(range(0, len(self.test_data), self.batch)):
            sources = self.test_data[i:i + self.batch]
            if self.beam_size > 1:
                ys = [self.model.translate(sources, self.max_length, beam=self.beam_size)]
            else:
                ys = [y.tolist() for y in self.model.translate(sources, self.max_length, beam=False)]
            hypotheses.extend(ys)
        return hypotheses


def main():
    args = get_translate_args()
    print(json.dumps(args.__dict__, indent=4))

    # Reading the vocab file
    with io.open(os.path.join(args.input, args.data + '.vocab.src.json'), encoding='utf-8') as f:
        source_id2w = json.load(f, cls=utils.Decoder)

    with io.open(os.path.join(args.input, args.data + '.vocab.trg.json'), encoding='utf-8') as f:
        target_id2w = json.load(f, cls=utils.Decoder)

    source_w2id = {w: i for i, w in source_id2w.items()}
    source_data = preprocess.make_dataset(args.src, source_w2id, args.tok)

    model = torch.load(args.model_file)
    hyp = TranslateText(model, source_data, batch=1, beam_size=args.beam_size)()
    save_output(hyp, target_id2w, args.output)


if __name__ == '__main__':
    main()
