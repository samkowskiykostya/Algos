import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

df = pd.read_csv('../../datasets/movie_data.csv')

# bag = CountVectorizer().fit_transform(df.review)
# raw_tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True).fit_transform(bag)
# print 'With stopwords: ', len([PorterStemmer().stem(w) for w in df.review[5].split()])
# print 'Without stopwords: ', len([PorterStemmer().stem(w) for w in df.review[5].split() if w not in stopwords.words('english')])

if __name__ == '__main__':
    stop = stopwords.words('english')

    X_train = df.loc[:25000, 'review'].values
    y_train = df.loc[:25000, 'sentiment'].values
    X_test = df.loc[25000:, 'review'].values
    y_test = df.loc[25000:, 'sentiment'].values
    param_grid = [{'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 {'vect__ngram_range': [(1,1)],
                   'vect__stop_words': [stop, None],
                   'vect__tokenizer': [tokenizer, tokenizer_porter],
                   'vect__use_idf':[False],
                   'vect__norm':[None],
                   'clf__penalty': ['l1', 'l2'],
                   'clf__C': [1.0, 10.0, 100.0]},
                 ]
    tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
    lr_tfidf = Pipeline([('vect', tfidf), ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='accuracy',
                               cv=5, verbose=1,
                               n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)
    print(('Best parameter set: %s ' % gs_lr_tfidf.best_params_))
    print(('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_))
    clf = gs_lr_tfidf.best_estimator_
    print(('Test Accuracy: %.3f' % clf.score(X_test, y_test)))