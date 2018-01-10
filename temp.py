from torchtext import data, datasets

src = data.Field()
trg = data.Field()

mt_train = datasets.TranslationDataset(path='data/', exts=('train-big.ja', 'train-big.en'), fields=(src, trg))

src.build_vocab(mt_train, max_size=40000)
trg.build_vocab(mt_train, max_size=40000)

mt_dev = datasets.TranslationDataset(path='data/', exts=('dev.ja', 'dev.en'), fields=(src, trg))

train_iter = data.BucketIterator(dataset=mt_train,
                                 batch_size=32,
                                 sort_key=lambda x: data.interleave_keys(len(x.src), len(x.trg)),
                                 device=-1)

next(iter(train_iter))