## Attention is All you Need (Transformer)

This repository implements the `transformer` model in *pytorch* framework which was introduced in the paper *[Attention is All you Need](https://arxiv.org/abs/1706.03762)* as described in their
NIPS 2017 version: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf


The overall model architecture is as shown in the figure:

![][transformer]

[transformer]: img/transformer.png "Transformer Model"


Run with this command:
python train.py -s train-big.ja -t train-big.en -g 0 -b 100 --tied --beam 5


BEST BLEU:
- 31.74 (1-layer model, B: 100, no Layer Norm)
- 32.09 (2-layer model, B: 80, no Layer Norm)
- 32.26 (1-layer model, B: 100)
- 31.63 (1-layer model, LayerNorm, data-bucketing, B: 100)
- 32.49 (1-layer model, B: 100, Beam Seach: 5)
- 32.81 (1-layer mode, B:128, Beam: 1)
- 34.65.(6-layer model, B:100, Beam: 5)

Without Tokenization
python train.py --tied -b 100 -g 0 --beam_size 5
BLEU: 0.3133, 0.638323/0.384471/0.269708/0.199853 (BP = 0.923720, ratio=0.93, hyp_len=4222, ref_len=4557)

python train.py --tied -b 128 -g 0 --beam_size 5
BLEU: 0.3291, 0.632548/0.392188/0.284585/0.213338 (BP = 0.939427, ratio=0.94, hyp_len=4289, ref_len=4557)

python train.py --tied -b 156 -g 0 --beam_size 5
BLEU: 0.3170, 0.604410/0.365872/0.259872/0.194633 (BP = 0.974893, ratio=0.98, hyp_len=4444, ref_len=4557)

python train.py --tied -b 128 -g 0 --beam_size 5 --pos_attention
BLEU: 0.3227, 0.622374/0.383763/0.269527/0.197917 (BP = 0.960395, ratio=0.96, hyp_len=4380, ref_len=4557)

python train.py --tied -b 100 -g 0 --beam_size 5 --pos_attention
BLEU: 0.3256, 0.620360/0.377281/0.271306/0.205811 (BP = 0.962901, ratio=0.96, hyp_len=4391, ref_len=4557)

python train.py --tied -b 100 -g 0 --beam_size 5 --pos_attention -d 0.3
BLEU: 0.3240, 0.627308/0.383586/0.269560/0.198476 (BP = 0.961990, ratio=0.96, hyp_len=4387, ref_len=4557)

Experimental:
- Beam Search

Steps to run the code

Preperation
python preprocess.py -s-train train-big.ja -t-train train-big.en


git checkout 78acbe019f91e2e41b1975e1a06e9519d66a48a4 [for best BLEU Scores]
Epoch  1,    50/ 1488; acc:   0.01; ppl: 24739.80; 22161 src tok/s; 16725 tgt tok/s;      3 s elapsed

#### Acknowledgements
* The code in this repository has been adapted from the [Sosuke Kobayashi](https://github.com/soskek)'s implementation in Chainer "https://github.com/soskek/attention_is_all_you_need".
* Some parts of the code were borrowed from [XNMT](https://github.com/neulab/xnmt/tree/master/xnmt) (based on [Dynet](https://github.com/clab/dynet)) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (based on [Pytorch](https://github.com/pytorch/pytorch)).