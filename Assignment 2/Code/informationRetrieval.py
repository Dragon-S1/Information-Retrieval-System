from util import *

# Add your import statements here
import math
import numpy as np



class InformationRetrieval():

	def __init__(self):
		self.index = None
		self.num_docs = None

	def buildIndex(self, docs, docIDs):
		"""
		Builds the document index in terms of the document
		IDs and stores it in the 'index' class variable

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is
			a document and each sub-sub-list is a sentence of the document
		arg2 : list
			A list of integers denoting IDs of the documents
		Returns
		-------
		None
		"""

		index = None

		#Fill in code here
		index = {}
		self.num_docs = len(docIDs)
		for docID in docIDs:
			doc = docs[docID-1]
			for sent in doc:
				for word in sent:
					if word in index.keys():
						if docID in index[word].keys():
							index[word][docID] += 1
						else:
							index[word][docID] = 1
					else:
						index[word] = {}
						index[word][docID] = 1

		self.index = index


	def rank(self, queries):
		"""
		Rank the documents according to relevance for each query

		Parameters
		----------
		arg1 : list
			A list of lists of lists where each sub-list is a query and
			each sub-sub-list is a sentence of the query
		

		Returns
		-------
		list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		"""

		doc_IDs_ordered = []

		#Fill in code here

		IDF = {word: math.log(self.num_docs/len(self.index[word]),2) for word in self.index.keys()}

		vecDs = []
		for docID in range(1, self.num_docs+1):
			vecD = {}
			for word in self.index.keys():
				if docID in self.index[word].keys():
					vecD[word] = self.index[word][docID]*IDF[word]
			vecDs.append(vecD)
		
		magDs = []

		for vecD in vecDs:
			magD = 0
			for word in vecD.keys():
				magD += vecD[word]*vecD[word]	
			magD = math.sqrt(magD)
			magDs.append(magD)

		for query in queries:
			ranking = []
			vecQ = {}
			for sent in query:
				for word in sent:
					if word in self.index.keys():
						if word in vecQ.keys():
							vecQ[word] += 1
						else:
							vecQ[word] = 1
			
			for word in vecQ.keys():
				vecQ[word] = vecQ[word]*IDF[word]

			magQ = 0
			for word in vecQ.keys():
				magQ += vecQ[word]*vecQ[word]
			magQ = math.sqrt(magQ)

			rel_docs = set()
			for word in vecQ.keys():
				for docID in self.index[word].keys():
					if self.index[word][docID] != 0 and vecQ[word] != 0:
						rel_docs.add(docID)
			
			rel_docs = np.array(list(rel_docs))
			cosines = []
			for doc in rel_docs:
				vecD = vecDs[doc-1]
				dot = 0
				for word in vecQ.keys():
					if word in vecD.keys():
						dot += vecQ[word]*vecD[word]

				cosine = dot / (magDs[doc-1] * magQ)
				cosines.append(cosine)
			idx = np.argsort(cosines)
			doc_IDs_ordered.append(list(rel_docs[idx][::-1]))
		return doc_IDs_ordered




