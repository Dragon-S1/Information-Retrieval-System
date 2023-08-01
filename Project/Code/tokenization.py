# Add your import statements here
import nltk.tokenize.treebank
import re

class Tokenization():

	def naive(self, text):
		"""
		Tokenization using a Naive Approach

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here
		tokenizedText = []
		for sent in text:
			# handle the cases where there is an apostrophe between a word
			contractions = re.compile(r"(\w+)('\w+)")
			cont_sent = contractions.sub(r"\1 \2", sent)
			
			# split the sentence string using punctuation marks and spaces
			# Also handles abbreviations
			tokenizedText.append([x for x in re.split(
				'([ ,;:()\{\}\[\]"?!#$&*+\/@=])|([.](?!\s*\w+))', cont_sent) if x != ' ' and x != '' and x != None])

		return tokenizedText



	def pennTreeBank(self, text):
		"""
		Tokenization using the Penn Tree Bank Tokenizer

		Parameters
		----------
		arg1 : list
			A list of strings where each string is a single sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of tokens
		"""

		tokenizedText = None

		#Fill in code here

		# use the treebank tokenizer from NLTK
		tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
		tokenizedText = tokenizer.tokenize_sents(text)

		return tokenizedText