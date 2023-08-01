from sentenceSegmentation import SentenceSegmentation
from tokenization import Tokenization
from inflectionReduction import InflectionReduction
from stopwordRemoval import StopwordRemoval
from informationRetrieval import InformationRetrieval
from evaluation import Evaluation
from weightingScheme import WeightingScheme

from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import anderson
from scipy.stats import wilcoxon

from sys import version_info
import argparse
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tqdm import tqdm

# Input compatibility for Python 2 and Python 3
if version_info.major == 3:
    pass
elif version_info.major == 2:
    try:
        input = raw_input
    except NameError:
        pass
else:
    print ("Unknown python version - input function not safe")


class SearchEngine:

	def __init__(self, args):
		self.args = args

		self.tokenizer = Tokenization()
		self.sentenceSegmenter = SentenceSegmentation()
		self.inflectionReducer = InflectionReduction()
		self.stopwordRemover = StopwordRemoval()

		self.informationRetriever = InformationRetrieval()
		self.evaluator = Evaluation()
		self.weightingScheme = WeightingScheme()
		self.models = ['VSM', 'LSI', 'BERT']
		self.weightingSchemes = ['Count', 'Basic', 'Normalized', 'Glasgow']
		self.modelData = {}

	def initData(self):
		self.modelData['VSM'] = {}
		self.modelData['LSI'] = {}
		self.modelData['BERT'] = None

		for scheme in self.weightingSchemes:
			self.modelData['VSM'][scheme] = None
			self.modelData['LSI'][scheme] = None

	def segmentSentences(self, text):
		"""
		Call the required sentence segmenter
		"""
		if self.args.segmenter == "naive":
			return self.sentenceSegmenter.naive(text)
		elif self.args.segmenter == "punkt":
			return self.sentenceSegmenter.punkt(text)

	def tokenize(self, text):
		"""
		Call the required tokenizer
		"""
		if self.args.tokenizer == "naive":
			return self.tokenizer.naive(text)
		elif self.args.tokenizer == "ptb":
			return self.tokenizer.pennTreeBank(text)

	def reduceInflection(self, text):
		"""
		Call the required stemmer/lemmatizer
		"""
		return self.inflectionReducer.reduce(text)

	def removeStopwords(self, text):
		"""
		Call the required stopword remover
		"""
		return self.stopwordRemover.fromList(text)


	def preprocessQueries(self, queries):
		"""
		Preprocess the queries - segment, tokenize, stem/lemmatize and remove stopwords
		"""

		# Segment queries
		segmentedQueries = []
		for query in queries:
			segmentedQuery = self.segmentSentences(query)
			segmentedQueries.append(segmentedQuery)
		json.dump(segmentedQueries, open(self.args.out_folder + "segmented_queries.txt", 'w'))
		# Tokenize queries
		tokenizedQueries = []
		for query in segmentedQueries:
			tokenizedQuery = self.tokenize(query)
			tokenizedQueries.append(tokenizedQuery)
		json.dump(tokenizedQueries, open(self.args.out_folder + "tokenized_queries.txt", 'w'))
		# Stem/Lemmatize queries
		reducedQueries = []
		for query in tokenizedQueries:
			reducedQuery = self.reduceInflection(query)
			reducedQueries.append(reducedQuery)
		json.dump(reducedQueries, open(self.args.out_folder + "reduced_queries.txt", 'w'))
		# Remove stopwords from queries
		stopwordRemovedQueries = []
		for query in reducedQueries:
			stopwordRemovedQuery = self.removeStopwords(query)
			stopwordRemovedQueries.append(stopwordRemovedQuery)
		json.dump(stopwordRemovedQueries, open(self.args.out_folder + "stopword_removed_queries.txt", 'w'))

		preprocessedQueries = stopwordRemovedQueries
		return preprocessedQueries

	def preprocessDocs(self, docs):
		"""
		Preprocess the documents
		"""
		
		# Segment docs
		segmentedDocs = []
		for doc in docs:
			segmentedDoc = self.segmentSentences(doc)
			segmentedDocs.append(segmentedDoc)
		json.dump(segmentedDocs, open(self.args.out_folder + "segmented_docs.txt", 'w'))
		# Tokenize docs
		tokenizedDocs = []
		for doc in segmentedDocs:
			tokenizedDoc = self.tokenize(doc)
			tokenizedDocs.append(tokenizedDoc)
		json.dump(tokenizedDocs, open(self.args.out_folder + "tokenized_docs.txt", 'w'))
		# Stem/Lemmatize docs
		reducedDocs = []
		for doc in tokenizedDocs:
			reducedDoc = self.reduceInflection(doc)
			reducedDocs.append(reducedDoc)
		json.dump(reducedDocs, open(self.args.out_folder + "reduced_docs.txt", 'w'))
		# Remove stopwords from docs
		stopwordRemovedDocs = []
		for doc in reducedDocs:
			stopwordRemovedDoc = self.removeStopwords(doc)
			stopwordRemovedDocs.append(stopwordRemovedDoc)
		json.dump(stopwordRemovedDocs, open(self.args.out_folder + "stopword_removed_docs.txt", 'w'))

		preprocessedDocs = stopwordRemovedDocs
		return preprocessedDocs

	def evaluation(self, data, query_ids, qrels):
		precisions, recalls, fscores, MAPs, nDCGs = [], [], [], [], []
		eval = {}
		header = ['k', 'Precision', 'Recall', 'F-score', 'MAP', 'nDCG']
		table = PrettyTable(header)
		for k in range(1, 11):
			precision = self.evaluator.meanPrecision(data, query_ids, qrels, k)
			precisions.append(precision)
			recall = self.evaluator.meanRecall(data, query_ids, qrels, k)
			recalls.append(recall)
			fscore = self.evaluator.meanFscore(data, query_ids, qrels, k)
			fscores.append(fscore)
			MAP = self.evaluator.meanAveragePrecision(data, query_ids, qrels, k)
			MAPs.append(MAP)
			nDCG = self.evaluator.meanNDCG(data, query_ids, qrels, k)
			nDCGs.append(nDCG)
			trunc = 4
			row = [k, float(f'{precision:.{trunc}f}'), float(f'{recall:.{trunc}f}'), float(f'{fscore:.{trunc}f}'), float(f'{MAP:.{trunc}f}'), float(f'{nDCG:.{trunc}f}')]
			table.add_row(row)

		eval['precision'] = precisions
		eval['recall'] = recalls
		eval['fscore'] = fscores
		eval['MAP'] = MAPs
		eval['nDCG'] = nDCGs

		return table, eval
	
	def plot(self, eval, model, scheme, k):
		# Plot the metrics and save plot 
			plt.plot(range(1, 11), eval['precision'], label="Precision")
			plt.plot(range(1, 11), eval['recall'], label="Recall")
			plt.plot(range(1, 11), eval['fscore'], label="F-Score")
			plt.plot(range(1, 11), eval['MAP'], label="MAP")
			plt.plot(range(1, 11), eval['nDCG'], label="nDCG")
			plt.xlabel("k")
			plt.legend()
			if model == 'BERT':
				plt.title(f"Evaluation Metrics - Cranfield Dataset - Model: {model}")
				plt.savefig(args.out_folder + f"eval_plot_{model}.png")
			elif model == 'VSM':
				plt.title(f"Evaluation Metrics - Cranfield Dataset - Model: {model} - Scheme: {scheme}")
				plt.savefig(args.out_folder + f"eval_plot_{model}_{scheme}.png")
			elif model == 'LSI':
				plt.title(f"Evaluation Metrics - Cranfield Dataset - Model: {model} - Scheme: {scheme}")
				plt.savefig(args.out_folder + f"eval_plot_{model}_{scheme}_{k}.png")
			plt.cla()

	def hypothesisTesting(self, query_ids, qrels, data1, data2):
		nDCG1 = []
		for query_id in query_ids:
			rel_docs = {}
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs[i["id"]] = 5-i["position"]
				nDCG1.append(self.evaluator.queryNDCG(data1[query_id-1], query_id, rel_docs, 10))

		nDCG2 = []
		for query_id in query_ids:
			rel_docs = {}
			for i in [x for x in qrels if x["query_num"] == str(query_id)]:
				rel_docs[i["id"]] = 5-i["position"]
				nDCG2.append(self.evaluator.queryNDCG(data2[query_id-1], query_id, rel_docs, 10))

		alternative = 'greater'  # Specify the desired direction ('greater' or 'less')
		statistic, p_value = wilcoxon(nDCG1, nDCG2, alternative=alternative)
		# print("Signed-Rank Test (one-sided -", alternative, "):")
		# print("Statistic:", statistic)
		# print("p-value:", p_value)
		better = 0
		if p_value < 0.05:
			better = 1
		elif p_value > 0.95:
			better = 2
		else:
			better = 0

		return statistic, p_value, better

	def evaluateDataset(self):
		"""
		- preprocesses the queries and documents, stores in output folder
		- invokes the IR system
		- evaluates precision, recall, fscore, nDCG and MAP 
		  for all queries in the Cranfield dataset
		- produces graphs of the evaluation metrics in the output folder
		"""

		for i in tqdm(range(1), desc="Read Queries"):
			# Read queries
			queries_json = json.load(open(args.dataset + "cran_queries.json", 'r'))[:]
			query_ids, queries = [item["query number"] for item in queries_json],[item["query"] for item in queries_json]
		
		for i in tqdm(range(1), desc="Process Queries"):
			# Process queries 
			processedQueries = self.preprocessQueries(queries)

		for i in tqdm(range(1), desc="Read Docs"):
			# Read documents
			docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
			doc_ids, docs = [item["id"] for item in docs_json], [item["body"] for item in docs_json]
		
		for i in tqdm(range(1), desc="Process Docs"):
			# Process documents
			processedDocs = self.preprocessDocs(docs)
		
		for i in tqdm(range(1), desc="Read Relevance Judements"):
			# Read relevance judements
			qrels = json.load(open(args.dataset + "cran_qrels.json", 'r'))[:]


		# Build document and query index
		for i in tqdm(range(1), desc="Build Index"):
			self.weightingScheme.build(processedDocs, doc_ids, processedQueries, query_ids)
		
		# provide informationRetriever class with required data
		self.informationRetriever.init(docs,doc_ids, queries, query_ids, processedDocs, processedQueries, self.weightingScheme.termList)
		
		# init model data
		self.initData()

		kValues = [200]
		# kValues = [10,50,100,200,300,400,600,700,800,1000,1200,1400]
		docWeights = {}
		queryWeights = {}

		# Generate Weight for different scheme
		for scheme in tqdm(self.weightingSchemes, desc = 'Building Weight Schemes'):
			docWeights[scheme], queryWeights[scheme] = self.weightingScheme.GenWeight(scheme)

		for model in tqdm(self.models, desc = 'Using Different Models'):
			# Rank the documents using VSM
			if model == 'VSM':
				for scheme in tqdm(self.weightingSchemes, desc = 'VSM for all Weight Schemes'):
					self.modelData['VSM'][scheme] = self.informationRetriever.informationRetrievalByVSM(docWeights[scheme], queryWeights[scheme])
			# Rank the documents using LSI
			elif model == 'LSI':
				for scheme in tqdm(self.weightingSchemes, desc = 'LSI for all Weight Schemes'):
					self.modelData['LSI'][scheme] = {}
					for k in tqdm(kValues, desc=f'LSI for different k values for {scheme} scheme'):
						self.informationRetriever.trainLSI(docWeights[scheme], k)
						self.modelData['LSI'][scheme][k] = self.informationRetriever.informationRetrievalByLSI(queryWeights[scheme])
			# Rank the documents using BERT
			elif model == 'BERT':
				for i in tqdm(range(1), desc='Building from BERT'):
					self.modelData['BERT'] = self.informationRetriever.informationRetrievalByBERT()

		header = ['Model-1', 'Model-2', 'Statistics', 'p_value', 'Result']
		testing = PrettyTable(header)

		for model1 in self.models:
			for model2 in self.models:
				if model1 == 'VSM':
					for scheme1 in self.weightingSchemes:
						if model2 == 'VSM':
							for scheme2 in self.weightingSchemes:
								if scheme1 != scheme2:
									statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1], self.modelData[model2][scheme2])
									res = ''
									if better == 1:
										res = f'{model1}({scheme1}) > {model2}({scheme2})'
									elif better == 2:
										res = f'{model1}({scheme1}) < {model2}({scheme2})'
									elif better == 0:
										res = f'{model1}({scheme1}) = {model2}({scheme2})'
									row = [f'{model1}({scheme1})', f'{model2}({scheme2})', statistics, p_value, res]
									testing.add_row(row)
						elif model2 == 'LSI':
							for scheme2 in self.weightingSchemes:
								for k in kValues:
									statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1], self.modelData[model2][scheme2][k])
									res = ''
									if better == 1:
										res = f'{model1}({scheme1}) > {model2}({scheme2})'
									elif better == 2:
										res = f'{model1}({scheme1}) < {model2}({scheme2})'
									elif better == 0:
										res = f'{model1}({scheme1}) = {model2}({scheme2})'
									row = [f'{model1}({scheme1})', f'{model2}({scheme2})', statistics, p_value, res]
									testing.add_row(row)
						elif model2 == 'BERT':
							statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1], self.modelData[model2])
							res = ''
							if better == 1:
								res = f'{model1}({scheme1}) > {model2}({scheme2})'
							elif better == 2:
								res = f'{model1}({scheme1}) < {model2}({scheme2})'
							elif better == 0:
								res = f'{model1}({scheme1}) = {model2}({scheme2})'
							row = [f'{model1}({scheme1})', f'{model2}({scheme2})', statistics, p_value, res]
							testing.add_row(row)
					# Rank the documents using LSI
				elif model1 == 'LSI':
					for scheme1 in self.weightingSchemes:
						for k1 in kValues:
							if model2 == 'VSM':
								for scheme2 in self.weightingSchemes:
									statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1][k1], self.modelData[model2][scheme2])
									res = ''
									if better == 1:
										res = f'{model1}({scheme1}) > {model2}({scheme2})'
									elif better == 2:
										res = f'{model1}({scheme1}) < {model2}({scheme2})'
									elif better == 0:
										res = f'{model1}({scheme1}) = {model2}({scheme2})'
									row = [f'{model1}({scheme1})', f'{model2}({scheme2})', statistics, p_value, res]
									testing.add_row(row)
							elif model2 == 'LSI':
								for scheme2 in self.weightingSchemes:
									if scheme1 != scheme2:
										for k2 in kValues:
											statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1][k1], self.modelData[model2][scheme2][k2])
											res = ''
											if better == 1:
												res = f'{model1}({scheme1}) > {model2}({scheme2})'
											elif better == 2:
												res = f'{model1}({scheme1}) < {model2}({scheme2})'
											elif better == 0:
												res = f'{model1}({scheme1}) = {model2}({scheme2})'
											row = [f'{model1}({scheme1})', f'{model2}({scheme2})', statistics, p_value, res]
											testing.add_row(row)
							elif model2 == 'BERT':
								statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1][scheme1][k1], self.modelData[model2])
								res = ''
								if better == 1:
									res = f'{model1}{scheme1} > {model2}'
								elif better == 2:
									res = f'{model1}{scheme1} < {model2}'
								elif better == 0:
									res = f'{model1}{scheme1} = {model2}'
								row = [f'{model1}({scheme1})', f'{model2}', statistics, p_value, res]
								testing.add_row(row)
			# Rank the documents using BERT
				elif model1 == 'BERT':
					if model2 == 'VSM':
						for scheme2 in self.weightingSchemes:
							statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1], self.modelData[model2][scheme2])
							res = ''
							if better == 1:
								res = f'{model1} > {model2}({scheme2})'
							elif better == 2:
								res = f'{model1} < {model2}({scheme2})'
							elif better == 0:
								res = f'{model1} = {model2}({scheme2})'
							row = [f'{model1}', f'{model2}({scheme2})', statistics, p_value, res]
							testing.add_row(row)
					elif model2 == 'LSI':
						for scheme2 in self.weightingSchemes:
							for k in kValues:
								statistics, p_value, better = self.hypothesisTesting(query_ids, qrels, self.modelData[model1], self.modelData[model2][scheme2][k])
								res = ''
								if better == 1:
									res = f'{model1} > {model2}({scheme2})'
								elif better == 2:
									res = f'{model1} < {model2}({scheme2})'
								elif better == 0:
									res = f'{model1} = {model2}({scheme2})'
								row = [f'{model1}', f'{model2}({scheme2})', statistics, p_value, res]
								testing.add_row(row)

		with open(self.args.out_folder + f"testing.txt", 'w') as f:
			f.write(str(testing))

		evaluate = True
		# Evaluate and Plot
		if evaluate:
			for model in tqdm(self.models, desc= 'Genrating Output'):
				if model == 'BERT':
					data = self.modelData['BERT']
					table, eval = self.evaluation(data, query_ids, qrels)
					self.plot(eval,model,'',0)
					with open(self.args.out_folder + f"{model}_eval.txt", 'w') as f:
						f.write(str(table))
				elif model == 'VSM':
					for scheme in self.weightingSchemes:
						data = self.modelData[model][scheme]		
						table, eval = self.evaluation(data, query_ids, qrels)
						self.plot(eval,model,scheme,0)
						with open(self.args.out_folder + f"{model}_{scheme}_eval.txt", 'w') as f:
							f.write(str(table))
				elif model == 'LSI':
					for scheme in self.weightingSchemes:
						for k in kValues:
							data = self.modelData[model][scheme][k]	
							table, eval = self.evaluation(data, query_ids, qrels)
							self.plot(eval,model,scheme,k)
							with open(self.args.out_folder + f"{model}_{scheme}_{k}_eval.txt", 'w') as f:
								f.write(str(table))

			# stat, p = shapiro(nDCG)
			# print('stat=%.3f, p=%.3f' % (stat, p))
			# if p > 0.05:
			# 	print('Probably Gaussian shapiro')
			# else:
			# 	print('Probably not Gaussian shapiro')

			# stat, p = normaltest(nDCG)
			# print('stat=%.3f, p=%.3f' % (stat, p))
			# if p > 0.05:
			# 	print('Probably Gaussian normaltest')
			# else:
			# 	print('Probably not Gaussian normaltest')
			# result = anderson(nDCG)
			# print('stat=%.3f' % (result.statistic))
			# for i in range(len(result.critical_values)):
			# 	sl, cv = result.significance_level[i], result.critical_values[i]
			# 	if result.statistic < cv:
			# 		print('Probably Gaussian anderson at the %.1f%% level' % (sl))
			# 	else:
			# 		print('Probably not Gaussian anderson at the %.1f%% level' % (sl))
		
	def handleCustomQuery(self):
		"""
		Take a custom query as input and return top five relevant documents
		"""

		#Get query
		print("Enter query below")
		query = input()

		queries = [query]
		query_ids = [1]

		# Process queries
		processedQueries = self.preprocessQueries(queries)

		# Read documents
		docs_json = json.load(open(args.dataset + "cran_docs.json", 'r'))[:]
		doc_ids, docs = [item["id"] for item in docs_json], \
							[item["body"] for item in docs_json]
		# Process documents
		processedDocs = self.preprocessDocs(docs)

		# Build document and query index
		for i in tqdm(range(1), desc="Build Index"):
			self.weightingScheme.build(processedDocs, doc_ids, processedQueries, query_ids)
		
		# provide informationRetriever class with required data
		self.informationRetriever.init(docs,doc_ids, queries, query_ids, processedDocs, processedQueries, self.weightingScheme.termList)
		
		# init model data
		self.initData()

		kValues = [200]
		# kValues = [10,50,100,200,300,400,600,700,800,1000,1200,1400]
		docWeights = {}
		queryWeights = {}
		
		# Rank the documents for the query
		for scheme in tqdm(self.weightingSchemes, desc = 'Building Weight Schemes'):
			docWeights[scheme], queryWeights[scheme] = self.weightingScheme.GenWeight(scheme)

		for model in tqdm(self.models, desc = 'Using Different Models'):
			# Rank the documents using VSM
			if model == 'VSM':
				for scheme in tqdm(self.weightingSchemes, desc = 'VSM for all Weight Schemes'):
					self.modelData['VSM'][scheme] = self.informationRetriever.informationRetrievalByVSM(docWeights[scheme], queryWeights[scheme])
			# Rank the documents using LSI
			elif model == 'LSI':
				for scheme in tqdm(self.weightingSchemes, desc = 'LSI for all Weight Schemes'):
					self.modelData['LSI'][scheme] = {}
					for k in tqdm(kValues, desc=f'LSI for different k values for {scheme} scheme'):
						self.informationRetriever.trainLSI(docWeights[scheme], k)
						self.modelData['LSI'][scheme][k] = self.informationRetriever.informationRetrievalByLSI(queryWeights[scheme])
			# Rank the documents using BERT
			elif model == 'BERT':
				for i in tqdm(range(1), desc='Building from BERT'):
					self.modelData['BERT'] = self.informationRetriever.informationRetrievalByBERT()

		# Print the IDs of first five documents
		print("\nTop ten document IDs using VSM : ")
		for id_ in self.modelData['VSM']['Basic'][0][:10]:
			print(id_)
		print("\nTop tem document IDs using LSI : ")
		for id_ in self.modelData['LSI']['Basic'][kValues[0]][0][:10]:
			print(id_)
		print("\nTop tem document IDs using BERT : ")
		for id_ in self.modelData['BERT'][0][:10]:
			print(id_)


if __name__ == "__main__":

	# Create an argument parser
	parser = argparse.ArgumentParser(description='main.py')

	# Tunable parameters as external arguments
	parser.add_argument('-dataset', default = "./cranfield/", 
						help = "Path to the dataset folder")
	parser.add_argument('-out_folder', default = "./output/", 
						help = "Path to output folder")
	parser.add_argument('-segmenter', default = "punkt",
	                    help = "Sentence Segmenter Type [naive|punkt]")
	parser.add_argument('-tokenizer',  default = "ptb",
	                    help = "Tokenizer Type [naive|ptb]")
	parser.add_argument('-custom', action = "store_true", 
						help = "Take custom query as input")
	
	# Parse the input arguments
	args = parser.parse_args()

	# Create an instance of the Search Engine
	searchEngine = SearchEngine(args)

	# Either handle query from user or evaluate on the complete dataset 
	if args.custom:
		searchEngine.handleCustomQuery()
	else:
		searchEngine.evaluateDataset()
