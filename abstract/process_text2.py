
from sklearn.externals import joblib
import pickle
from sklearn.neighbors import KNeighborsClassifier
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

#online trfidf
cur = abstract_db.find({},{'abstract':1})
hash_vect= HashingVectorizer(stop_words='english', ngram_range=(1,3), non_negative=True, norm=None)
list_of_rows = []
y=[]
for doc in cur:
    sparse_row = hash_vect.fit_transform(doc['abstract'])
    list_of_rows.append(sparse_row)
    y.append(str(doc['_id']))
hash_stack = vstack(list_of_rows)
print "ithink i can"
abstract_db.create_index('author_without_collab_display')
print "ithink i can"
h_transform = TfidfTransformer()
hash_matrix = h_transform.fit_transform(hash_stack)

#export ids, and first 2 processing steps
pickle.dump(y,open("ids.pkl","wb"))
joblib.dump(hash_vect,'step_1.pkl')
joblib.dump(h_transform,'step_2.pkl')
print "ithink i can"

select_feature =  VarianceThreshold(threshold=.00001)
reduced_matrix = select_feature.fit_transform(hash_matrix)
joblib.dump(select_feature,'step_3.pkl')
print "ithink i can"
# make a random sample to fit pca to
random_sample = resample(reduced_matrix,replace =False, n_samples =10000)
cosine_pca= KernelPCA(n_components=1000,kernel='cosine')
cosine_pca.fit(random_sample)
topics = cosine_pca.transform(reduced_matrix)
print "ithink i can"

joblib.dump(cosine_pca,'step_4.pkl')
pickle.dump(topics,open("topics.pkl","wb"))
print "ithink i can"
kn =KNeighborsClassifier(algorithm= 'brute',metric='cosine')
kn.fit(topics,y)
joblib.dump(kn,'kn_search.pkl')
print "i did"