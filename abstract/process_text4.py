
#This pics up right before the kernel pca is fit
from sklearn.externals import joblib
import pickle
import numpy as np 
from bson.objectid import ObjectId
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import KernelPCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from scipy.sparse import vstack
from sklearn.utils import resample
print "just getting started"
#starting up pymongo                                                            
import pymongo
#making mongo db of abstracts                                                   
c = pymongo.MongoClient()
db = c['PloS']
abstract_db = db['abstracts2']
print "ithink i can"




step_1 =joblib.load('step_1.pkl')
step_2 =joblib.load('step_2.pkl')
step_3 =joblib.load('step_3.pkl')


#online trfidf transformation with out trainng                                                                 
cur = abstract_db.find({},{'abstract':1})

list_of_rows = []

for doc in cur:
    sparse_row = step_1.transform(doc['abstract'])
    list_of_rows.append(sparse_row)
    
hash_stack = vstack(list_of_rows)


print "ithink i can"

print "ithink i can"
h_transform = step_2
hash_matrix = h_transform.transform(hash_stack)


print "ithink i can"

select_feature =  step_3
reduced_matrix = select_feature.transform(hash_matrix)

print "ithink i can"
# make a random sample to fit pca to                                            
step_4 = joblib.load('step_4.pkl')
cosine_pca= step_4

del hash_stack
del hash_matrix

del list_of_rows
del h_transform
del select_feature

print "this shit just got real!"
#importing multi proc
from multiprocessing import Pool
#make a list of ranges
rngs=[]
chunk=100
for i in range(0,reduced_matrix.shape[0],chunk):
    if i + chunk < reduced_matrix.shape[0]:
        r =(i,i+chunk)
        rngs.append(r)
    else:
        r= (i,None)
        rngs.append(r)

#defining function i will map
def pare_trans(x):
    return cosine_pca.transform(reduced_matrix[x[0]:x[1]])

p = Pool(4)
print "ithink i can transform"
foo = p.map(pare_trans, rngs)
vfoo = np.vstack(foo)

print "ithink i can"


pickle.dump(vfoo,open("topics.pkl","wb"))

print "i did"