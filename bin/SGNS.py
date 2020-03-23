# -*- coding:utf-8 -*-
#! usr/bin/env python3
"""
Created on 20/03/2020 下午9:48 
@Author: xinzhi 
"""

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import argparse
import os

def main(args: str):

    if args.pre_fix != '':
        embedding = os.path.join(args.out_path, '{0}_embedding.txt'.format(args.pre_fix))
        embedding_bin = os.path.join(args.out_path, '{0}_embedding.bin'.format(args.pre_fix))
    else:
        embedding = os.path.join(args.out_path, 'embedding.txt')
        embedding_bin = os.path.join(args.out_path, 'embedding.bin')

    print('-'*50)
    print('Training.')
    model = Word2Vec(LineSentence(args.corpus_file), size=args.size, window=args.window, min_count=args.min_count,
                     workers=args.workers, sg=args.sg, hs=args.hs, negative=args.negative, ns_exponent=args.ns_exponent,
                     cbow_mean=args.cbow_mean, alpha=args.alpha, min_alpha=args.min_alpha, seed=args.seed, max_vocab_size=args.max_vocab_size,
                     max_final_vocab=args.max_final_vocab, sample=args.sample, iter=args.iter, sorted_vocab=args.sorted_vocab, batch_words=args.batch_words)
    model.save(embedding_bin)
    model.wv.save_word2vec_format(embedding, binary=False)
    print('Embedding save done. {0}'.format(embedding))
    print('-'*50)

def word2vec_config():
    """
    class gensim.models.word2vec.Word2Vec(sentences=None, corpus_file=None, size=100, alpha=0.025, window=5, min_count=5,
    max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, sg=0, hs=0, negative=5, ns_exponent=0.75,
    cbow_mean=1, hashfxn=<built-in function hash>, iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000,
    compute_loss=False, callbacks=(), max_final_vocab=None)

    sentence: the sentences iterable can be simply a list of lists of tokens.
    corpus_file: Path to ac orpus file in LineSentence format.
    size: Dimensionality of the word vectors.
    window: Maximum distance between the current and predicted word within a sentence.
    min_count: Ignores all words with total frequency lower than this.
    workers: Use these many worker threads to train the model (=faster training with multicore machines).
    sg: Training algorithm: 1 for skip-gram; otherwise CBOW.
    hs: If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.
    negative: If > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20).
    ns_exponent: The exponent used to shape the negative sampling distribution.
    cbow_mean ({0, 1}, optional) – If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.
    alpha (float, optional) – The initial learning rate.
    min_alpha (float, optional) – Learning rate will linearly drop to min_alpha as training progresses.
    seed (int, optional) – Seed for the random number generator. Initial vectors for each word are seeded with a hash of the concatenation of word + str(seed).
    max_vocab_size (int, optional) – Limits the RAM during vocabulary building;
    max_final_vocab (int, optional) – Limits the vocab to a target vocab size by automatically picking a matching min_count.
    sample (float, optional) – The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).
    hashfxn (function, optional) – Hash function to use to randomly initialize weights, for increased training reproducibility.
    iter (int, optional) – Number of iterations (epochs) over the corpus.
    trim_rule (function, optional) –
    Vocabulary trimming rule, specifies whether certain words should remain in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).

    Documentation: https://radimrehurek.com/gensim/models/word2vec.html
    """
    parser = argparse.ArgumentParser(description='Knowledge Graph Embedding tunable configs.')

    ''' Input and output config'''
    parser.add_argument('-cp', dest='corpus_file', type=str, required=True)
    parser.add_argument('-op', dest='out_path', type=str, default='../data/embedding')
    parser.add_argument('-pf', dest='pre_fix', type=str, default='pto')

    ''' Word2Vec configs '''
    parser.add_argument('-sz', dest='size', type=int, default=100)
    parser.add_argument('-wd', dest='window', type=int, default=5)
    parser.add_argument('-mc', dest='min_count', type=int, default=5)
    parser.add_argument('-wk', dest='workers', type=int, default=3)
    parser.add_argument('-sg', dest='sg', choices=[0, 1], type=int, default=0, help='Training algorithm: 1 for skip-gram; otherwise CBOW.')
    parser.add_argument('-hs', dest='hs', choices=[0, 1], type=int, default=0, help='If 1, hierarchical softmax will be used for model training. If 0, and negative is non-zero, negative sampling will be used.')
    parser.add_argument('-ng', dest='negative', type=int, default=5, help='f > 0, negative sampling will be used, the int for negative specifies how many “noise words” should be drawn (usually between 5-20). If set to 0, no negative sampling is used.')
    parser.add_argument('-ns', dest='ns_exponent', type=float, default=0.75, help='The exponent used to shape the negative sampling distribution. ')
    parser.add_argument('-cm', dest='cbow_mean', choices=[0, 1], type=int, default=1, help='If 0, use the sum of the context word vectors. If 1, use the mean, only applies when cbow is used.')
    parser.add_argument('-al', dest='alpha', type=float, default=0.025, help='The initial learning rate.')
    parser.add_argument('-ma', dest='min_alpha', type=float, default=0.0001, help='Learning rate will linearly drop to min_alpha as training progresses.')
    parser.add_argument('-se', dest='seed', type=int, default=1, help='Seed for the random number generator.')
    parser.add_argument('-ms', dest='max_vocab_size', type=int, default=None, help='Limits the RAM during vocabulary building')
    parser.add_argument('-mf', dest='max_final_vocab', type=int, default=None, help='Limits the vocab to a target vocab size by automatically picking a matching min_count.')
    parser.add_argument('-sa', dest='sample', type=float, default=0.001, help=' The threshold for configuring which higher-frequency words are randomly downsampled, useful range is (0, 1e-5).')
    parser.add_argument('-it', dest='iter', type=int, default=5)
    parser.add_argument('-sv', dest='sorted_vocab', choices=[0, 1], default=1)
    parser.add_argument('-bw', dest='batch_words', type=int, default=10000)

    return parser.parse_args()


if __name__ == '__main__':
    
    """
    if you want to test this code:
     python SGNS.py -cp 'data/abs_corpus/pub_1' -sg 1 -hs 0
    """
    
    args = word2vec_config()

    main(args)

