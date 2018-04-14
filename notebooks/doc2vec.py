import pickle
import numpy as np
import sys
import gensim
import os
import collections
import smart_open
import random

import csv
plist = "./playlist.txt"

print (gensim.models.doc2vec.FAST_VERSION)
with open(plist) as inputfile:
	document = [i[:-1].split(' ') for i in inputfile.readlines()]
	print ("finished...")
	    
	artist_title_lookup = []
	tfidf_corpus = []

	def progress(cnt, total, div):
	    if cnt % div == 0:
	        print ("Preprocessing... ", cnt / total*1.0 * 100, '%')
	    
	def read_corpus(playlist, tokens_only=False):
	    for i, doc in enumerate(playlist):
	        progress(i, len(playlist), 60000)
	        if tokens_only:
	            yield gensim.utils.simple_preprocess(doc)
	        else:
	            yield gensim.models.doc2vec.TaggedDocument(doc, [i])

	corpus = list(read_corpus(document))
	print ("Finished")
	# print (corpus[:2])
	# TRAINING...
	model_dict = dict()
	epoch = 20 # Set to 2 because slow

	print ("BUILDING DOC2VEC...")
	model = gensim.models.doc2vec.Doc2Vec(vector_size=100, min_count=3, epochs=epoch, workers = 8)
	model.build_vocab(corpus)
	model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
	model_dict = model
	    
	print ("DOC2VEC FINISHED BUILDING...SAVING...")
	f_save = open('model.p','wb')
	pickle.dump(model_dict,f_save)
	f_save.close()
	print ('MODEL SAVED...')