#import search engine                                                                                                                       
#from searcher_2 import *                                                                                                                   
# set up search engine                                                                                                                      
# this should all be in the search package i'm importing but fuck it
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
from sklearn.utils import resample
import numpy as np


#starting up pymongo
import pymongo
#making mongo db of abstracts
c = pymongo.MongoClient()
db = c['PloS']
abstract_db = db['this_year']

doc_ids = pickle.load(open("ids.pkl","rb"))
topics = pickle.load(open("topics.pkl","rb"))
#neigh= joblib.load('kn_search.pkl')
step_1 =joblib.load('step_1.pkl')
step_2 =joblib.load('step_2.pkl')
step_3 =joblib.load('step_3.pkl')
step_4 = joblib.load('step_4.pkl')


from nearpy import Engine
from nearpy.hashes import  RandomBinaryProjectionTree, RandomBinaryProjections
from nearpy.filters import NearestFilter
from nearpy.distances import CosineDistance
# need to import something to cosine distance
dimension =1000

def transform_text(text):
    """takes the text and turns it into 1000 dimentional vecotr"""
    text_vector =step_4.transform(step_3.transform(step_2.transform(step_1.transform([text]))))[0]
    return text_vector

import time

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
            v= topics[i]
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
                       
            
            
                

def about_auth(name, comp_text ="marryhad a little lamb"):
    """this takes an authors name and an abstract its being compared to,
    this then gives all the data needed for thee summary page"""
    docs = list(abstract_db.find({"author_without_collab_display":name}))
    compvect=transform_text(comp_text)
    atopoics=Set()
    coauths =Set()
    journals =Set()
    total_views=0
    score_index =[]
    articles =[]
    abscore=[]
    total_articles =len(docs)
    for i, doc in enumerate(docs):
        sim = cosine_similarity(compvect,doc['vector'])[0][0]
        score_index.append((i,sim))
        atopoics.update(doc['subject_level_1'])
        coauths.update(doc['author_display'])
        journals.update([doc['journal']])
        total_views+= doc['counter_total_all']
        abscore.extend(doc['abstract'])
        article ={}
        article['title_display']=doc['title_display']
        article['publication_date']=doc['publication_date']
        article['author_display']=doc['author_display']
        article['counter_total_all']=doc['counter_total_all']
        article['sim']=sim
        article['abstract']=doc['abstract']
        articles.append(article)
        

    score_index =sorted(score_index, key=lambda x:x[1],reverse=True)   
    coauths.remove(name)
    return atopoics,coauths,journals, total_views,total_articles, score_index,articles





search1 = searcher(128,20,N=20)                                                                                                            

import flask
import urlparse
# #starting up pymongo
# import pymongo
# #making mongo db of abstracts
# c = pymongo.MongoClient()
# db = c['PloS']
# abstract_db = db['this_year']

app = flask.Flask(__name__)

@app.route("/")
def homepage():
    """This should be the home page where you can put in you abstract                                                                       
to search"""
    return flask.render_template('cool.html')
  

@app.route("/search", methods = ["POST"])
def search():
    """Search results page (for now only showing the query)"""
    query = urlparse.parse_qs(flask.request.get_data())
    
    authors_to_ex = query['au'][0].split(', ')
    recs = search1.relevent_authors(query['q'][0], TOP=10,authors_to_exclude=authors_to_ex)
    # doc =abstract_db.find_one({"author_without_collab_display":query['au'][0]})
    # return flask.render_template('results.html',
    #                              abstract = query['q'][0],
    #                              author = query['au'][0])
    # return flask.render_template('results.html',
    #                              abstract = query['q'][0],
    #                              author = query['au'][0],
    #                              ab2 = doc['abstract'][0])
    return flask.render_template('results.html',
                                 abstract = query['q'][0],
                                 author = query['au'][0],
                                 recs = recs)
 
@app.route("/author/<author_name>")
def show_author_page(author_name):
    """make a page with all the authros abstracts"""
    atopoics,coauths,journals, total_views,total_articles, score_index,articles=about_auth(author_name)
    #doc =abstract_db.find({"author_without_collab_display":author_name})
    return flask.render_template('author.html',
                                author=author_name,
                                topics = atopoics,
                                coauths= coauths,
                                journals= journals,
                                total_articles=total_articles,
                                total_views=total_views,
                                keywords = None,
                                articles =articles)


if __name__ == '__main__':

    app.debug=True
    app.run(host='0.0.0.0')

