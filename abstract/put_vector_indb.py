from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sets import Set
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.decomposition import PCA, KernelPCA
from sklearn.feature_selection import VarianceThreshold
from bson.objectid import ObjectId
import numpy as np
#starting up pymongo
import pymongo
#making mongo db of abstracts
c = pymongo.MongoClient()
db = c['PloS']
abstract_db = db['abstracts2']



doc_ids = pickle.load(open("ids.pkl","rb"))
topics = pickle.load(open("topics.pkl","rb"))
for i, doc_id in enumerate(doc_ids):
    abstract_db.update({"_id":ObjectId(doc_id)}, {"$set":{"vector":topics[i][0].tolist()}},False,True) # the 0 index is becasue i did the pac row by row