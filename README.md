# Attention_is_All_You_Need

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

Experimental:
- Beam Search