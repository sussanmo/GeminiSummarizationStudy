import nltk
nltk.download('punkt')
print(nltk.data.find('tokenizers/punkt/english.pickle'))


from nltk.tokenize import word_tokenize

text = "This is a test sentence."
tokens = word_tokenize(text)
print(tokens)
