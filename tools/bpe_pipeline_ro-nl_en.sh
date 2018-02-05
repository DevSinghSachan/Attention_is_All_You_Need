#!/usr/bin/env bash
# Author : Thamme Gowda
# Created : Nov 06, 2017

TF="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

#======= EXPERIMENT SETUP ======

# update these variables
NAME="run_ro-nl_en"
OUT="tf-runs/$NAME"

DATA="/projects/tir1/users/dsachan/Attention_is_All_You_Need/data/ro-nl_en"
DATA_L1="/projects/tir1/users/dsachan/Attention_is_All_You_Need/data/ro_en"
DATA_L2="/projects/tir1/users/dsachan/Attention_is_All_You_Need/data/nl_en"

TRAIN_SRC=$DATA/train.ro-nl
TRAIN_TGT=$DATA/train.en
VALID_SRC=$DATA/dev.ro-nl
VALID_TGT=$DATA/dev.en
TEST_SRC=$DATA/test.ro-nl
TEST_TGT=$DATA/test.en

TEST_SRC_L1=$DATA_L1/test.ro
TEST_TGT_L1=$DATA_L1/test.en

TEST_SRC_L2=$DATA_L2/test.nl
TEST_TGT_L2=$DATA_L2/test.en

BPE="" # default
BPE="src+tgt" # src, tgt, src+tgt

# applicable only when BPE="src" or "src+tgt"
BPE_SRC_OPS=32000

# applicable only when BPE="tgt" or "src+tgt"
BPE_TGT_OPS=32000

GPUARG="" # default
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
    echo "BPE on source"
    # Here we could use more  monolingual data
    $TF/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_SRC > $OUT/data/bpe-codes.src

    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TRAIN_SRC > $OUT/data/train.src
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $VALID_SRC > $OUT/data/valid.src
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC > $OUT/data/test.src

    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC_L1 > $OUT/data/test_l1.src
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.src <  $TEST_SRC_L2 > $OUT/data/test_l2.src

else
    ln -sf $TRAIN_SRC $OUT/data/train.src
    ln -sf $VALID_SRC $OUT/data/valid.src
    ln -sf $TEST_SRC $OUT/data/test.src
fi


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE on target"
    # Here we could use more  monolingual data
    $TF/tools/learn_bpe.py -s $BPE_SRC_OPS < $TRAIN_TGT > $OUT/data/bpe-codes.tgt

    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $TRAIN_TGT > $OUT/data/train.tgt
    $TF/tools/apply_bpe.py -c $OUT/data/bpe-codes.tgt <  $VALID_TGT > $OUT/data/valid.tgt

    # We dont touch the test References, No BPE on them!
    ln -sf $TEST_TGT $OUT/data/test.tgt
    ln -sf $TEST_TGT_L1 $OUT/data/test_l1.tgt
    ln -sf $TEST_TGT_L2 $OUT/data/test_l2.tgt
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
--batchsize 128 --tied --beam 5 --dropout 0.2 --epoch 40 --layers 1 --multi_heads 8 --gpu 0 \
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

echo "Step 3a: Translate Test"
# Translate L1
python $TF/translate.py -i $OUT/data --data processed --batchsize 128 --beam_size 5 \
--model_file $model --src $OUT/data/test_l1.src --output $OUT/test/test_l1.out

# Translate L2
python $TF/translate.py -i $OUT/data --data processed --batchsize 128 --beam_size 5 \
--model_file $model --src $OUT/data/test_l2.src --output $OUT/test/test_l2.out


if [[ "$BPE" == *"tgt"* ]]; then
    echo "BPE decoding/detokenising target to match with references"
    mv $OUT/test/test.out{,.bpe}
    mv $OUT/test/valid.out{,.bpe}
    mv $OUT/test/test_l1.out{,.bpe}
    mv $OUT/test/test_l2.out{,.bpe}

    cat $OUT/test/valid.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/valid.out
    cat $OUT/test/test.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test.out

    cat $OUT/test/test_l1.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test_l1.out
    cat $OUT/test/test_l2.out.bpe | sed -E 's/(@@ )|(@@ ?$)//g' > $OUT/test/test_l2.out

fi

echo "Step 4a: Evaluate Test"
perl $TF/tools/multi-bleu.perl $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.tc.bleu
perl $TF/tools/multi-bleu.perl $OUT/data/test_l1.tgt < $OUT/test/test_l1.out > $OUT/test/test_l1.tc.bleu
perl $TF/tools/multi-bleu.perl $OUT/data/test_l2.tgt < $OUT/test/test_l2.out > $OUT/test/test_l2.tc.bleu


perl $TF/tools/multi-bleu.perl -lc $OUT/data/test.tgt < $OUT/test/test.out > $OUT/test/test.lc.bleu

echo "Step 4b: Evaluate Dev"
perl $TF/tools/multi-bleu.perl $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.tc.bleu
perl $TF/tools/multi-bleu.perl -lc $OUT/data/valid.tgt < $OUT/test/valid.out > $OUT/test/valid.lc.bleu

#===== EXPERIMENT END ======
