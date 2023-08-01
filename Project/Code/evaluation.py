# Add your import statements here
import math

class Evaluation():

	def queryPrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The precision value as a number between 0 and 1
		"""

		precision = -1

		#Fill in code here
		precision = len([x for x in query_doc_IDs_ordered[:k] if str(x) in true_doc_IDs])/k
		return precision


	def meanPrecision(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of precision of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean precision value as a number between 0 and 1
		"""

		meanPrecision = -1

		#Fill in code here
		meanPrecision = 0
		for query_id in query_ids:
			rel_docs = []
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs.append(i["id"])
			meanPrecision += self.queryPrecision(doc_IDs_ordered[query_id-1], query_id, rel_docs, k)
		
		meanPrecision = meanPrecision/len(query_ids)
		return meanPrecision

	
	def queryRecall(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The recall value as a number between 0 and 1
		"""

		recall = -1

		#Fill in code here
		recall = len([x for x in query_doc_IDs_ordered[:k] if str(x) in true_doc_IDs])/len(true_doc_IDs)
		return recall


	def meanRecall(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of recall of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean recall value as a number between 0 and 1
		"""

		meanRecall = -1

		#Fill in code here
		meanRecall = 0
		for query_id in query_ids:
			rel_docs = []
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs.append(i["id"])
			meanRecall += self.queryRecall(doc_IDs_ordered[query_id-1], query_id, rel_docs, k)
		
		meanRecall = meanRecall/len(query_ids)
		return meanRecall


	def queryFscore(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The fscore value as a number between 0 and 1
		"""

		fscore = -1

		#Fill in code here

		precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		recall = self.queryRecall(query_doc_IDs_ordered, query_id, true_doc_IDs, k)
		fscore = 0
		if precision != 0 and recall != 0:
			fscore = 2*precision*recall/(precision+recall)
		return fscore


	def meanFscore(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of fscore of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value
		
		Returns
		-------
		float
			The mean fscore value as a number between 0 and 1
		"""

		meanFscore = -1

		#Fill in code here
		meanFscore = 0
		for query_id in query_ids:
			rel_docs = []
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs.append(i["id"])
			meanFscore += self.queryFscore(doc_IDs_ordered[query_id-1], query_id, rel_docs, k)

		meanFscore = meanFscore/len(query_ids)
		return meanFscore
	

	def queryNDCG(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at given value of k for a single query

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of IDs of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The nDCG value as a number between 0 and 1
		"""

		nDCG = -1

		#Fill in code here
		DCG = 0
		IDCG = 0
		sort_pos = []
		for i in true_doc_IDs.keys():
			sort_pos.append(true_doc_IDs[i])
		for i in range(0,k-len(sort_pos)):
			sort_pos.append(0)

		sort_pos.sort(reverse=True)

		for i in range(1, k+1):
			log2 = math.log(i+1,2)
			rel = 0
			if str(query_doc_IDs_ordered[i-1]) in true_doc_IDs.keys():
				rel = true_doc_IDs[str(query_doc_IDs_ordered[i-1])]
			DCG += rel/log2
			IDCG += sort_pos[i-1]/log2
		
		nDCG = DCG/IDCG
		return nDCG

	def meanNDCG(self, doc_IDs_ordered, query_ids, qrels, k):
		"""
		Computation of nDCG of the Information Retrieval System
		at a given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries for which the documents are ordered
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The mean nDCG value as a number between 0 and 1
		"""

		# meanNDCG = -1

		#Fill in code here
		meanNDCG = 0
		for query_id in query_ids:
			rel_docs = {}
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs[i["id"]] = 5-i["position"]
			meanNDCG += self.queryNDCG(doc_IDs_ordered[query_id-1], query_id, rel_docs, k)

		meanNDCG = meanNDCG/len(query_ids)

		return meanNDCG


	def queryAveragePrecision(self, query_doc_IDs_ordered, query_id, true_doc_IDs, k):
		"""
		Computation of average precision of the Information Retrieval System
		at a given value of k for a single query (the average of precision@i
		values for i such that the ith document is truly relevant)

		Parameters
		----------
		arg1 : list
			A list of integers denoting the IDs of documents in
			their predicted order of relevance to a query
		arg2 : int
			The ID of the query in question
		arg3 : list
			The list of documents relevant to the query (ground truth)
		arg4 : int
			The k value

		Returns
		-------
		float
			The average precision value as a number between 0 and 1
		"""

		avgPrecision = -1

		#Fill in code here
		avgPrecision = 0
		for i in range(1,k+1):
			precision = self.queryPrecision(query_doc_IDs_ordered, query_id, true_doc_IDs, i)
			rel = 1 if str(query_doc_IDs_ordered[i-1]) in true_doc_IDs else 0
			avgPrecision += precision * rel
		avgPrecision = avgPrecision/len(true_doc_IDs)
		return avgPrecision


	def meanAveragePrecision(self, doc_IDs_ordered, query_ids, q_rels, k):
		"""
		Computation of MAP of the Information Retrieval System
		at given value of k, averaged over all the queries

		Parameters
		----------
		arg1 : list
			A list of lists of integers where the ith sub-list is a list of IDs
			of documents in their predicted order of relevance to the ith query
		arg2 : list
			A list of IDs of the queries
		arg3 : list
			A list of dictionaries containing document-relevance
			judgements - Refer cran_qrels.json for the structure of each
			dictionary
		arg4 : int
			The k value

		Returns
		-------
		float
			The MAP value as a number between 0 and 1
		"""

		meanAveragePrecision = -1

		#Fill in code here
		meanAveragePrecision = 0
		for query_id in query_ids:
			rel_docs = []
			for i in [x for x in q_rels if x["query_num"] == str(query_id)]:
				rel_docs.append(i["id"])
			meanAveragePrecision += self.queryAveragePrecision(doc_IDs_ordered[query_id-1], query_id, rel_docs, k)

		meanAveragePrecision = meanAveragePrecision/len(query_ids)
		return meanAveragePrecision

