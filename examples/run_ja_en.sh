AIAYN="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

cd ..

# Preprocess
python ${AIAYN}/preprocess.py -i ${AIAYN}/data/ja_en -s-train train-big.ja -t-train train-big.en -s-valid dev.ja -t-valid dev.en -s-test test.ja -t-test test.en --save_data demo

# Train
python ${AIAYN}/train.py -i ${AIAYN}/data/ja_en --data demo --batchsize 128 --tied --beam 5 --dropout 0.2 --epoch 40 --layer 1 --multi_heads 8 --gpu 0 --model_file "${AIAYN}/results/model.ckpt"
