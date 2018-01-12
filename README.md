# Attention_is_All_You_Need

Run with this command:
python train.py -s train-big.ja -t train-big.en -g 0 -b 100 --tied --beam 5


BEST BLEU:
- 31.74 (1-layer model, B: 100, no Layer Norm)
- 32.09 (2-layer model, B: 80, no Layer Norm)
- 32.26 (1-layer model, B: 100, Layer Norm)
- 31.63 (1-layer model, LayerNorm, data-bucketing, B: 100)
- 32.49 (1-layer model, B: 100, Layer Norm, Beam Seach: 5)

Experimental:
- Beam Search