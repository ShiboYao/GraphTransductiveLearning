
from scipy import sparse
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer



def get20news(voc_size = None):
    '''
    cate = ['alt.atheism', 
            'talk.religion.misc', 
            'comp.graphics', 
            'sci.space']#default 4 groups, otherwise vectors have too many dimensions
    '''
    cate = ['rec.autos',
            'rec.motorcycles',
            'rec.sport.baseball',
            'rec.sport.hockey']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=cate)
    #newsgroups_train = fetch_20newsgroups(subset='train')
    vectorizer = TfidfVectorizer(max_features=voc_size)
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    labels = newsgroups_train.target
    labels = labels.reshape(-1,1)

    data = sparse.hstack((vectors,labels))
    data = data.tocsr()

    return data
