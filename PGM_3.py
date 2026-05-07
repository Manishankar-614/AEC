import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

# Download required data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = "The quick brown foxes are jumping over the lazy dogs."

# Preprocessing
words = word_tokenize(text.lower())
words = [word for word in words if word.isalpha() and word not in stopwords.words('english')]

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(word) for word in words]

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(word) for word in words]

# Output
print("Original Words:", words)
print("Stemmed Words:", stemmed)
print("Lemmatized Words:", lemmatized)