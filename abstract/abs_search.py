from sklearn.externals import joblib
import pickle

from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.sparse import vstack
from nearpy import Engine
from nearpy.hashes import  RandomBinaryProjectionTree
import time

#starting up pymongo
import pymongo
#making mongo db of abstracts
c = pymongo.MongoClient()
db = c['PloS']
abstract_db = db['abstracts2']

doc_ids = pickle.load(open("ids.pkl","rb"))
topics = pickle.load(open("topics.pkl","rb"))
step_1 =joblib.load('step_1.pkl')
step_2 =joblib.load('step_2.pkl')
step_3 =joblib.load('step_3.pkl')
step_4 = joblib.load('step_4.pkl')


def transform_text(text):
    """takes the text and turns it into 1000 dimentional vecotr"""
    text_vector =step_4.transform(step_3.transform(step_2.transform(step_1.transform([text]))))[0]
    return text_vector

