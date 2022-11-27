import re
import nltk
import numpy as np
import unicodedata
import pandas as pd

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# Reading data
data = pd.read_csv('tmdb_5000_movies.csv')
data.head(10)


# Let's also take care of the null values present in the data
data.fillna('', inplace = True)

# Text Processing
STOPWORDS = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def process_text(text):

  text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
  text = re.sub(r'[^a-zA-Z\s]', '', text)
  #text = text.translate(str.maketrans('', '', string.punctuation))
  text = text.lower()
  text = " ".join([word for word in str(text).split() if word not in STOPWORDS])
  text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

  return text

# Applying the preprocessing on the overview column of the dataset
data['processed_overview'] = df2['overview'].map(process_text)

# Also, we shall select the top 4 columns for our problem statement
data = data[['title', 'overview', 'processed_overview', 'tagline']]

# First let us get the processed data 
data_list = data['processed_overview'].to_list()

# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(min_df = 0., max_df = 1.)
count_vect_matrix = count_vect.fit_transform(data_list)
print(count_vect_matrix.shape)
# Output - (4803, 20449)

# CountVectorized features' similarity matrix
from sklearn.metrics.pairwise import cosine_similarity

count_doc_sim = cosine_similarity(count_vect_matrix)
# Let us create a dataframe out of this matrix for easy retrieval of data
count_doc_sim_df = pd.DataFrame(count_doc_sim)
count_doc_sim_df.head()

# Now let's retrieve some data from this matrix
# Getting index from a movie title
movies = data['title'].to_list()
movie_idx = movies.index("Captain America: Civil War")
print(movie_idx)

# Getting the specific row from the similarity matrix (dataframe)
movie_similarities = count_doc_sim_df.iloc[movie_idx].values
print(movie_similarities)

# Getting the Top-5 similar movies' indices
similar_movie_idxs = np.argsort(-movie_similarities)[1:6]
print(similar_movie_idxs)

# Getting the movie title's using the indices 
similar_movies = []
for i in similar_movie_idxs:
    similar_movies.append(movies[i])

print(similar_movies)

# Creating a function which will take a similarity matrix and movie title as input and get the top 5 recommended titles
def get_similar_document(movie_title, similarity_matrix):
    
    index = movies.index(movie_title)
    sim = sim_matrix[index].values
    sim_index = np.argsort(-sim)[1:6]
    
    similar_movies = []
    for i in sim_index:
        similar_movies.append(movies[i])
    
    return similar_movies
  
# Now it will be easy to get the similar_docs given a title and the similarity matrix
get_similar_document("Captain America: Civil War", count_doc_sim_df)

# Moving onto the Tf-Idf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tf_idf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf_idf.fit_transform(data_list)
print(tfidf_matrix.shape)

from sklearn.metrics.pairwise import cosine_similarity

tf_doc_sim = cosine_similarity(tfidf_matrix)
tf_doc_sim_df = pd.DataFrame(tf_doc_sim)
tf_doc_sim_df.head()

get_recommendations("Captain America: Civil War", tf_doc_sim_df)
