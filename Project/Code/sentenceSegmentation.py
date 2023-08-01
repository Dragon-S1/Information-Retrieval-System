# Add your import statements here
import nltk.tokenize.punkt
import re

class SentenceSegmentation():

	def naive(self, text):
		"""
		Sentence Segmentation using a Naive Approach

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each string is a single sentence
		"""

		segmentedText = None

		#Fill in code here

		# segmenting the string using regex to find match till a punctuation mark is encountered
		segmentedText = [x.lstrip() for x in re.findall('[^.?!]+[.?!]?', text)]		

		return segmentedText





	def punkt(self, text):
		"""
		Sentence Segmentation using the Punkt Tokenizer

		Parameters
		----------
		arg1 : str
			A string (a bunch of sentences)

		Returns
		-------
		list
			A list of strings where each strin is a single sentence
		"""

		segmentedText = None

		#Fill in code here

		# use the pre-trained punkt tokenizer
		tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
		segmentedText = tokenizer.tokenize(text, realign_boundaries=False)
		
		return segmentedText