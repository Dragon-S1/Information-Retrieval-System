# Add your import statements here
from nltk.stem.snowball import SnowballStemmer

class InflectionReduction:

	def reduce(self, text):
		"""
		Stemming/Lemmatization

		Parameters
		----------
		arg1 : list
			A list of lists where each sub-list a sequence of tokens
			representing a sentence

		Returns
		-------
		list
			A list of lists where each sub-list is a sequence of
			stemmed/lemmatized tokens representing a sentence
		"""

		reducedText = None

		#Fill in code here

		# use stemmer from NLTK package
		stemmer = SnowballStemmer(language='english')
		reducedText = []
		for sent in text:
			reducedText.append([stemmer.stem(word) for word in sent])
		
		return reducedText


