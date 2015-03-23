from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
import pickle

from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sets import Set
from sklearn.metrics import pairwise_distances
from collections import Counter
from sklearn.decomposition import  KernelPCA
from sklearn.feature_selection import VarianceThreshold
from bson.objectid import ObjectId
import numpy as np
import time
from sklearn.utils import resample
from nearpy import Engine
from nearpy.hashes import  RandomBinaryProjectionTree, RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import CosineDistance

#starting up pymongo
import pymongo
#making mongo db of abstracts
c = pymongo.MongoClient()d
db = c['PloS']
abstract_db = db['abstracts2']

doc_ids = pickle.load(open("ids.pkl","rb"))
topics = pickle.load(open("topics.pkl","rb"))

step_1 =joblib.load('step_1.pkl')
step_2 =joblib.load('step_2.pkl')
step_3 =joblib.load('step_3.pkl')
step_4 = joblib.load('step_4.pkl')

dimension =1000

def transform_text(text):
    """takes the text and turns it into 1000 dimentional vecotr"""
    text_vector =step_4.transform(step_3.transform(step_2.transform(step_1.transform([text]))))[0]
    return text_vector

class searcher:
    """This builds a nearpy search engine.  With the vectorized abstracts
    once compleated the object should have methods that allow finding 
    the top x autors based on a text query, it should also should have 
    some methods for testing the quality of its search results."""
    dimension =1000 #dimensions of vectorized text
    
    #init will build the nearpy engine
    def __init__(self,projection_count,minimum_result_size,N=20):
        """init will build the nearpy engine with RandomBinaryProjectionsTree
        with the parameters specified byt the projection_count and minimum_result_size, 
        N closest results."""
        self.projection_count = projection_count
        self.minimum_result_size=minimum_result_size
        self.N = N
        
        self.rbp = RandomBinaryProjectionTree('rbp', 
                                              projection_count,
                                              minimum_result_size)
        engine =Engine(dimension,
                       lshashes=[self.rbp],
                       vector_filters=[NearestFilter(self.N)], distance=CosineDistance())
        for i in range(len(doc_ids)):
            v= topics[i][0] # the 0 index is because the topics are a lis of arrays becasue i had to do pca one row at a time
            d= doc_ids[i]
            engine.store_vector(v, data = d)
        self.engine =  engine
    
    def query_(self,text):
        """this is for internal use of the nearpy engine to find the mongo db document ids 
        for similar abstracts. it retruns a list of results, each result has the vectorized text of an abstract
        the id numbers, and the distance"""
        return self.engine.neighbours(transform_text(text))
    
    def get_similar_abs(self,text):
        """returns the N closets documents to your abstract text. """ 
    
    
        text_vector =transform_text(text)
        q_res = self.query_(text)
        results ={}
        authors = Set()
        subjects =Set()
        for i in range(self.N):
            doc = abstract_db.find_one({"_id":ObjectId(q_res[i][1])})
            results[doc['title_display']]={}
            results[doc['title_display']]['dist']=q_res[i][2]
            results[doc['title_display']]['authors']= doc['author_without_collab_display']
            results[doc['title_display']]['views']= doc['counter_total_all']
            results[doc['title_display']]['subjects']= doc['subject_level_1']
            authors.update(doc['author_without_collab_display'])
            subjects.update(doc['subject_level_1'])
            #results[doc['title_display']]['abstract']= doc['abstract']
        return authors, results, subjects, text_vector
    
    
    
    def relevent_authors(self, text, TOP=10,match=1, authors_to_exclude=[], ids_to_exclude=[], smooth =100):
        """TOP returns the relevent authors, match is how much you want to weigh having a good match
        you can spesify a set of authors or document ids you want to exclude from you results"""
    
        text_vector =transform_text(text)
        q_res = self.query_(text)
        q_ids=Set()
        for i in range(len(q_res)):
            q_ids.update([q_res[i][1]])
           
        excluded_ids =Set()
        for i in ids_to_exclude:
            excluded_ids.update([i])
        q_ids.difference_update(excluded_ids)
        authors = Set()
        for i in q_ids:
            doc = abstract_db.find_one({"_id":ObjectId(i)})
            authors.update(doc['author_without_collab_display'])
        
        auth_score=[]
        times =[] #for timing vectorization
        
        a = authors.difference(Set(authors_to_exclude))

        for author in a:
            cursor = abstract_db.find({"author_without_collab_display":author})
            author_vects=[]
            author_views=smooth
            for i in cursor:
                author_vects.append(i['vector'])
                author_views+=i['counter_total_all']
            start= time.time()
            average_vector =np.asarray(author_vects).mean(axis=0)
            end = time.time()
        
            #print author, author_views
            #print cosine_similarity(average_vector,text_vector)
            sim = cosine_similarity(average_vector,text_vector)[0][0]**match
            auth_score.append((author,sim*author_views))
            times.append(start-end)
        top_auths = sorted(auth_score, key=lambda x:x[1],reverse=True)
        return top_auths[0:TOP-1] #, times
    
    def relevent_authors2(self, text, TOP=10,match=1, authors_to_exclude=[], ids_to_exclude=[], smooth=1):
        """TOP returns the relevent authors, match is how much you want to weigh having a good match
        you can spesify a set of authors or document ids you want to exclude from you results"""
    
        text_vector =transform_text(text)
        q_res = self.query_(text)
        q_ids=Set()
        for i in range(len(q_res)):
            q_ids.update([q_res[i][1]])
           
        excluded_ids =Set()
        for i in ids_to_exclude:
            excluded_ids.update([i])
        q_ids.difference_update(excluded_ids)
        authors = Set()
        for i in q_ids:
            doc = abstract_db.find_one({"_id":ObjectId(i)})
            authors.update(doc['author_without_collab_display'])
        
        auth_score=[]
        times =[] #for timing vectorization
        
        a = authors.difference(Set(authors_to_exclude))

        for author in a:
            cursor = abstract_db.find({"author_without_collab_display":author})
            author_sims =[]
            for i in cursor:
                author_vect=np.asarray(i['vector'])
                author_views=i['counter_total_all']+smooth
                sim = ((cosine_similarity(text_vector,author_vect)[0][0])*author_views)**match
                author_sims.append(sim)
            start= time.time()
            
            end = time.time()
        
            #print author, author_views
            #print cosine_similarity(average_vector,text_vector)
            #sim = cosine_similarity((average_vector,text_vector)[0][0])**match
            auth_score.append((author,np.asarray(author_sims).mean(axis=0)))
            times.append(start-end)
        top_auths = sorted(auth_score, key=lambda x:x[1],reverse=True)
        return top_auths[0:TOP-1] #, times
    
    def test(self, tests=1, TOP=10,match=1):
        """test the search functionality by randomly choosing articles from the db, and seeing how many of the authors show up
        in the top search results"""
        test_ids = resample(doc_ids,n_samples =tests)
        q_times=[]
        q_scores=[]
        for t_id in doc_ids:
            test_authors = Set()
            doc = abstract_db.find_one({"_id":ObjectId(t_id)})
            test_authors.update(doc['author_without_collab_display'])
            abstract_text = doc['abstract'][0]
            start= time.time()
            test_results = self.relevent_authors(abstract_text, TOP, match, ids_to_exclude=[t_id])
            stop =time.time()
            top_10=Set()
            for x,y in test_results:
                top_10.update([x])
            q_times.append(start-stop)
            score= len(test_authors.intersection(top_10))/len(test_authors)
            q_scores.append(score)
        return q_times, q_scores

def timer(function,**kwargs):
    """times a function, ** kwargs are the arguments the function takes"""
    start= time.time()
    results = function(**kwargs)
    end = time.time()
    return end-start, results


def get_test_text():
    """randomly grabs an abstract text"""
    sample_id = resample(doc_ids,n_samples =1)[0]
    sample_doc = abstract_db.find_one({"_id":ObjectId(sample_id)})
    text= sample_doc['abstract'][0]
    authors=sample_doc['author_without_collab_display']
    title =sample_doc['title_display']
    return text, authors, sample_id, title