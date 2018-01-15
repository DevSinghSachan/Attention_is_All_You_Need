# Preprocess the dataset
python preprocess.py -i data/en_vi -s-train train.en -t-train train.vi -s-valid tst2012.en -t-valid tst2012.vi -s-test tst2013.en -t-test tst2013.vi


# Train the model
python train.py --tied -b 100 -g 0 --beam_size 5 --report_every 50 -i data/en_vi --data demo