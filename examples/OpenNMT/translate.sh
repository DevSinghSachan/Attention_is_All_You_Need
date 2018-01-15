python translate.py -model models_ja_en_acc_63.38_ppl_9.24_e29.pt -src data/dev.ja -output pred.txt \
-verbose -report_bleu -gpu 0 -beam_size 5 -tgt data/dev.en -alpha 1




# (Transformer) Result for ja->en train-big dataset included with XNMT. evaluated on dev set
# [Beam: 5, alpha 1] || BLEU = 28.36, 62.8/37.1/25.3/18.5 (BP=0.877, ratio=0.884, hyp_len=4030, ref_len=4557)
# [Beam: 5, alpha 0] || BLEU ~(approximately) 27.71
# [Beam: 1, alpha 0] || BLEU ~(approximately) 26.83

# (BiLSTM encoder) Result for ja->en train-big dataset included with XNMT. evaluated on dev set
# [Beam: 5, alpha 0.2] BLEU = 31.19, 62.2/37.2/25.9/19.1 (BP=0.953, ratio=0.954, hyp_len=4347, ref_len=4557)
# [Beam: 1, alpha 0.0] BLEU = 29.55, 59.1/34.3/23.2/16.2 (BP=1.000, ratio=1.018, hyp_len=4640, ref_len=4557)
#


