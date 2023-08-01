import math

class WeightingScheme():
	def __init__(self):
		self.docIndex = None
		self.queryIndex = None
		self.num_docs = None
		self.num_terms = None
		self.docIDs = None
		self.queryIDs = None
		self.termList = None
		self.IDF = None
		self.docLength = None
		self.queryLength = None

	def build(self, docs, docIDs, queries, queryIDs):
		docIndex = {}
		queryIndex = {}
		self.docIDs = docIDs
		self.queryIDs = queryIDs
		self.num_docs = len(docIDs)
		docLength = {}
		queryLength = {}
		terms = set()
		
		for docID in docIDs:
			doc = docs[docID-1]
			doc_length = set()
			for sent in doc:
				for term in sent:
					doc_length.add(term)
					terms.add(term)
					if term in docIndex.keys():
						if docID in docIndex[term].keys():
								docIndex[term][docID] += 1
						else:
								docIndex[term][docID] = 1
					else:
						docIndex[term] = {}
						docIndex[term][docID] = 1
			docLength[docID] = len(doc_length)

		for queryID in queryIDs:
			query = queries[queryID-1]
			query_length = set()
			for sent in query:
				for term in sent:
					query_length.add(term)
					if term in docIndex.keys():
						if term in queryIndex.keys():
							if queryID in queryIndex[term].keys():
									queryIndex[term][queryID] += 1
							else:
									queryIndex[term][queryID] = 1
						else:
							queryIndex[term] = {}
							queryIndex[term][queryID] = 1
			queryLength[queryID] = len(query_length)

		self.docIndex = docIndex
		self.queryIndex = queryIndex
		self.termList = list(terms)
		self.num_terms = len(self.termList)
		self.IDF = {term: math.log(self.num_docs/len(self.docIndex[term]),2) for term in self.docIndex.keys()}
		self.docLength = docLength
		self.queryLength = queryLength

	def WeightList(self):
		D = {}
		Q = {}
		for term in self.termList:
			D[term] = {}
			Q[term] = {}
			for doc in self.docIDs:
				D[term][doc] = 0
			for query in self.queryIDs:
				Q[term][query] = 0
		return D,Q
	
	def GenWeight(self, name):
		D = {}
		Q = {}
		if name == 'Count':
			D, Q = self.Count()
		elif name == 'Basic':
			D, Q = self.Basic()
		elif name == 'Normalized':
			D, Q = self.Normalized()
		elif name == 'Glasgow':
			D, Q = self.Glasgow()
		
		return D, Q

	def Count(self):
		D, Q = self.WeightList()

		for term in self.termList:
			for doc in self.docIDs:
				if doc in self.docIndex[term].keys():
					D[term][doc] = self.docIndex[term][doc]
			if term in self.queryIndex.keys():
				for query in self.queryIDs:
					if query in self.queryIndex[term].keys():
						Q[term][query] = self.queryIndex[term][query]

		return D, Q

	def Basic(self):
		D, Q = self.WeightList()

		for term in self.termList:
			for doc in self.docIDs:
				if doc in self.docIndex[term].keys():
					D[term][doc] = self.docIndex[term][doc]*self.IDF[term]
			if term in self.queryIndex.keys():
				for query in self.queryIDs:
					if query in self.queryIndex[term].keys():
						Q[term][query] = self.queryIndex[term][query]*self.IDF[term]

		return D, Q


	def Normalized(self):
		D, Q = self.WeightList()

		docMaxFreq = {}
		queryMaxFreq = {}
        
		for doc in self.docIDs:
			max_freq = 0
			for term in self.docIndex.keys():
					if doc in self.docIndex[term].keys():
							max_freq = max(max_freq,self.docIndex[term][doc])
			docMaxFreq[doc] = max_freq

		for query in self.queryIDs:
			max_freq = 0
			for term in self.queryIndex.keys():
					if query in self.queryIndex[term].keys():
							max_freq = max(max_freq,self.queryIndex[term][query])
			queryMaxFreq[query] = max_freq

		for term in self.termList:
			for doc in self.docIDs:
				if doc in self.docIndex[term].keys():
					D[term][doc] = (self.docIndex[term][doc]/docMaxFreq[doc])*self.IDF[term]
			if term in self.queryIndex.keys():
				for query in self.queryIDs:
					if query in self.queryIndex[term].keys():
						Q[term][query] = (0.5+0.5*self.queryIndex[term][query]/queryMaxFreq[query])*self.IDF[term]
	  
		return D, Q


	def Glasgow(self):
		D, Q = self.WeightList()

		for term in self.termList:
			for doc in self.docIDs:
				if doc in self.docIndex[term].keys():
					D[term][doc] = (math.log(self.docIndex[term][doc]+1,2)/math.log(self.docLength[doc],2))*self.IDF[term]
			if term in self.queryIndex.keys():
				for query in self.queryIDs:
					if query in self.queryIndex[term].keys():
						Q[term][query] = (math.log(self.queryIndex[term][query]+1,2)/math.log(self.queryLength[query],2))*self.IDF[term]

		return D, Q
