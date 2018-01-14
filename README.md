## Attention is All you Need (Transformer)

This repository implements the `transformer` model in *pytorch* framework which was introduced in the paper *[Attention is All you Need](https://arxiv.org/abs/1706.03762)* as described in their
NIPS 2017 version: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf


The overall model architecture is as shown in the figure:

![][transformer]

[transformer]: img/transformer.png "Transformer Model"


The code in this repository implements the following features:
* Positional Encoding
* Multi-Head Dot-Product Attention
* Positional Attention from "*[Non-Autoregressive Neural Machine Translation](https://arxiv.org/abs/1711.02281)*"
* Label Smoothing
* Warm-up steps based training of Adam Optimizer
* LayerNorm and residual connections after each sublayer
* Shared weights of target embedding and decoder softmax layer
* Beam Search (Experimental)

## Software Requirements
* Python 3.6
* Pytorch v0.4 (needs manual installation from source https://github.com/pytorch/pytorch)
* torchtext
* numpy

One can install the above packages using the requirements file.

## Usage

### Step 1: Preprocessing:
`python preprocess.py -i data/ja_en -s-train train-big.ja -t-train train-big.en -s-valid dev.ja -t-valid dev.en --save_data demo`

### Step 2: Train and Evaluate the model:
`python train.py --data demo -g 0 -b 128 --tied --beam 5 -d 0.2 --epoch 40 --layer 1 --multi_heads 8`


## Results

BLEU Scores on Ja->En translation task with various configurations:
- **31.33** (Layers=1, B=100, Beam=5)
BLEU: 31.33, 63.8/38.4/26.9/19.9 (BP = 0.923, ratio=0.93, hyp_len=4222, ref_len=4557)
- **32.91** (Layers=1, B=128, Beam=5)
BLEU: 32.91, 63.2/39.2/28.4/21.3 (BP = 0.939, ratio=0.94, hyp_len=4289, ref_len=4557)
- **31.70** (Layers=1, B=156, Beam=5)
BLEU: 31.70, 60.4/36.6/25.9/19.5 (BP = 0.975, ratio=0.98, hyp_len=4444, ref_len=4557)
- **32.56** (Layers=1, B=100, Beam=5, Pos_Attention=True)  
BLEU: 32.56, 62.0/37.7/27.1/20.6 (BP = 0.963, ratio=0.96, hyp_len=4391, ref_len=4557)
- **34.65** (Layers=6, B=100, Beam=5)


## Training Speed:
For about 150K training examples on the above dataset, the model takes approximately 60 seconds for 1 epoch on a modern Titan-Xp GPU with 12GB RAM.


[//]: <> (git checkout 78acbe019f91e2e41b1975e1a06e9519d66a48a4 , "eval" branch, for best BLEU Scores)

## Acknowledgements
* Thanks for the suggestions from Graham Neubig @gneubig and Matt Sperber @msperber
* The code in this repository was originally based and has been adapted from the [Sosuke Kobayashi](https://github.com/soskek)'s implementation in Chainer "https://github.com/soskek/attention_is_all_you_need".
* Some parts of the code were borrowed from [XNMT](https://github.com/neulab/xnmt/tree/master/xnmt) (based on [Dynet](https://github.com/clab/dynet)) and [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) (based on [Pytorch](https://github.com/pytorch/pytorch)).