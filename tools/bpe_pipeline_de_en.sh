#!/usr/bin/env bash

TF="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======

# update these variables
NAME="run_de_en"
OUT="tf-runs/$NAME"

DATA="/projects/tir1/users/dsachan/Attention_is_All_You_Need/data/de_en"
TRAIN_SRC=$DATA/train.de
TRAIN_TGT=$DATA/train.en
TEST_SRC=$DATA/test.de
TEST_TGT=$DATA/test.en

head -500 $DATA/dev.de > $DATA/dev.small.de
head -500 $DATA/dev.en > $DATA/dev.small.en
VALID_SRC=$DATA/dev.small.de
VALID_TGT=$DATA/dev.small.en

BPE="src+tgt" # src, tgt, src+tgt
BPE_OPS=20000
GPUARG="0"

#====== EXPERIMENT BEGIN ======

# Check if input exists
for f in $TRAIN_SRC $TRAIN_TGT $VALID_SRC $VALID_TGT $TEST_SRC $TEST_TGT; do
    if [[ ! -f "$f" ]]; then
        echo "Input File $f doesnt exist. Please fix the paths"
        exit 1
    fi
done

function lines_check {
    l1=`wc -l $1`
    l2=`wc -l $2`
    if [[ $l1 != $l2 ]]; then
        echo "ERROR: Record counts doesnt match between: $1 and $2"
        exit 2
    fi
}
# lines_check $TRAIN_SRC $TRAIN_TGT
# lines_check $VALID_SRC $VALID_TGT
# lines_check $TEST_SRC $TEST_TGT


echo "Output dir = $OUT"
[ -d $OUT ] || mkdir -p $OUT
[ -d $OUT/data ] || mkdir -p $OUT/data
[ -d $OUT/models ] || mkdir $OUT/models
[ -d $OUT/test ] || mkdir -p  $OUT/test


echo "Step 1a: Preprocess inputs"

if [[ "$BPE" == *"src"* ]]; then
    echo "Learning BPE on source and target combined"
    cat ${TRAIN_SRC} ${TRAIN_TGT} | $TF/tools/learn_bpe.py -s ${BPE_OPS} > $OUT/data/bpe-codes.${BPE_OPS}
fi


if [[ "$BPE" == *"src"* ]]; then
    echo "Applying BPE on source"
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_SRC > $OUT/data/train.src
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_SRC > $OUT/data/valid.src
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.${BPE_OPS} <  $TEST_SRC > $OUT/data/test.src
else
    ln -sf $TRAIN_SRC $OUT/data/train.src
    ln -sf $VALID_SRC $OUT/data/valid.src
    ln -sf $TEST_SRC $OUT/data/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "Applying BPE on target"
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.${BPE_OPS} <  $TRAIN_TGT > $OUT/data/train.tgt
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.${BPE_OPS} <  $VALID_TGT > $OUT/data/valid.tgt

    # We dont touch the test References, No BPE on them!
    ln -sf $TEST_TGT $OUT/data/test.tgt
else
    ln -sf $TRAIN_TGT $OUT/data/train.tgt
    ln -sf $VALID_TGT $OUT/data/valid.tgt
    ln -sf $TEST_TGT $OUT/data/test.tgt
fi


#: <<EOF
echo "Step 1b: Preprocess"
python ${TF}/preprocess.py \
      -i ${OUT}/data \
      -s-train train.src \
      -t-train train.tgt \
      -s-valid valid.src \
      -t-valid valid.tgt \
      -s-test test.src \
      -t-test test.tgt \
      --save_data processed

echo "Step 2: Train"
GPU_OPTS=""
if [[ ! -z $GPUARG ]]; then
    GPU_OPTS="-gpuid $GPUARG"
fi

CMD="python $TF/train.py -i $OUT/data --data processed --model_file $OUT/models/model_$NAME.ckpt --data processed \
--batchsize 128 --tied --beam 5 --dropout 0.2 --epoch 40 --layers 6 --multi_heads 8 --gpu 0 \
--dev_hyp $OUT/test/valid.out --test_hyp $OUT/test/test.out"

echo "Training command :: $CMD"
eval "$CMD"

#EOF

# select a model with high accuracy and low perplexity
model=$OUT/models/model_$NAME.ckpt
echo "Chosen Model = $model"
if [[ -z "$model" ]]; then
    echo "Model not found. Looked in $OUT/models/"
    exit 1
fi

GPU_OPTS=""
if [ ! -z $GPUARG ]; then
    GPU_OPTS="-gpu $GPUARG"
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/test.out{,.bpe}
    mv $OUT/test/valid.out{,.bpe} 
    cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
    cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out
fi

echo "Step 4a: Evaluate Test"
perl $TF/tools/multi-bleu.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
perl $TF/tools/multi-bleu.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
