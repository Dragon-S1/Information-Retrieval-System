# Add your import statements here
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

class InformationRetrieval():

	def __init__(self):
		self.docs = None
		self.queries = None
		self.docIDs = None
		self.queryIDs = None
		self.processedDocs = None
		self.processedQueries = None
		self.termList = None
		#LSI
		self.termsRepInKDim = None
		self.docsRepInKDim = None

	def init(self, docs, docIDs, queries, queryIDs, processedDocs, processedQueries, termList):
		self.docs = docs
		self.queries = queries
		self.docIDs = docIDs
		self.queryIDs = queryIDs
		self.processedDocs = processedDocs
		self.processedQueries = processedQueries
		self.termList = termList

	def informationRetrievalByVSM(self, DocWeights, QueryWeights):
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

		vecDs = []
		for docID in self.docIDs:
			vecD = {}
			for term in DocWeights.keys():
				if DocWeights[term][docID] != 0:
					vecD[term] = DocWeights[term][docID]
			vecDs.append(vecD)
		
		magDs = []

		for vecD in vecDs:
			magD = 0
			for term in vecD.keys():
				magD += vecD[term]*vecD[term]	
			magD = math.sqrt(magD)
			magDs.append(magD)

		for queryID in self.queryIDs:
			vecQ = {}
			
			for term in QueryWeights.keys():
				if QueryWeights[term][queryID] != 0:
					vecQ[term] = QueryWeights[term][queryID]

			magQ = 0
			for term in vecQ.keys():
				magQ += vecQ[term]*vecQ[term]
			magQ = math.sqrt(magQ)

			rel_docs = set()
			for term in vecQ.keys():
				for docID in self.docIDs:
					if DocWeights[term][docID] != 0:
						rel_docs.add(docID)
			
			rel_docs = np.array(list(rel_docs))
			cosines = []

			for doc in rel_docs:
				vecD = vecDs[doc-1]
				dot = 0
				for term in vecQ.keys():
					if term in vecD.keys():
						dot += vecQ[term]*vecD[term]

				cosine = dot / (magDs[doc-1] * magQ)
				cosines.append(cosine)
			idx = np.argsort(cosines)[::-1]
			doc_IDs_ordered.append(list(rel_docs[idx]))

		return doc_IDs_ordered
	
	def trainLSI(self, DocWeights, k):
		termDocMatrix = np.zeros((len(self.termList), len(self.docIDs)))

		# Fill the tf-idf values
		for i in range(len(termDocMatrix)):
			for j in range(len(termDocMatrix[i])):
				termDocMatrix[i,j] = DocWeights[self.termList[i]][self.docIDs[j]]

		# Perform Singular Value Decomposition
		u, s, vh = np.linalg.svd(termDocMatrix)
		Sig = np.diag(s)
		self.termsRepInKDim = u[:,:k]@Sig[:k,:k]
		self.docsRepInKDim = vh.T[:,:k]@Sig[:k,:k]

	def informationRetrievalByLSI(self, QueryWeights):
		termList = self.termList
		termIndexMap = {}

		for i in range(len(termList)):
			termIndexMap[termList[i]] = i

		doc_IDs_ordered = []

		for query in self.queryIDs:
			queryVector = np.zeros(len(self.termList))

			for i in range(len(queryVector)):
				queryVector[i] = QueryWeights[termList[i]][query]

			queryRepInKDim = queryVector@self.termsRepInKDim

			cosine_similarities = cosine_similarity(queryRepInKDim.reshape(1, -1), self.docsRepInKDim)
			nearest_indices = cosine_similarities[0].argsort()[::-1]
			doc_IDs_ordered.append(list(nearest_indices))
		return doc_IDs_ordered

	def informationRetrievalByBERT(self):
		docs = self.docs
		queries = self.queries
		doc_IDs_ordered = []

		model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
		# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

		re_encode = False

		doc_enc = []
		if not os.path.exists('./encode/docs.pt') or re_encode:
			for j in tqdm(range(len(docs)), desc ="Encoding Doc"):
					doc_enc.append(model.encode(docs[j]))
			torch.save(doc_enc, './encode/docs.pt')
		else:
			doc_enc = torch.load('./encode/docs.pt')

		query_enc = []
		if not os.path.exists('./encode/queries.pt') or re_encode:
			for i in tqdm(range(len(queries)), desc ="Encoding Queries"):
					query_enc.append(model.encode(queries[i]))
			torch.save(query_enc, './encode/queries.pt')
		else:
			query_enc = torch.load('./encode/queries.pt')
		
		cos_sim = cosine_similarity(query_enc, doc_enc)

		for cos_similarity_vector in cos_sim:
				top_rank = (cos_similarity_vector.argsort()[::-1])+1
				doc_IDs_ordered.append(top_rank)

		return doc_IDs_ordered
