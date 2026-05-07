import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

# Download required data
nltk.download('punkt')
nltk.download('stopwords')

text = "This is a sample text. It is used for testing the NLTK word frequency counting functionality."

# Tokenize words and remove stopwords
words = word_tokenize(text.lower())
words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]

# Count frequency
freq = Counter(words)

# Print result
print("Word Frequencies:")
for word, count in freq.items():
    print(word, ":", count)