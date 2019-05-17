import re

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.corpus import stopwords


def stream_docs(path):
    with(open(path,'r')) as csv:
        next(csv)
        for line in csv:
            text,label = line[:-3], int(line[-2])
            yield text, label
def get_minibatch(stream, size):
    docs,y=[],[]
    try:
        for _ in range(size):
            text,label=next(stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None,None
    return docs,y
def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

stop = stopwords.words('english')
vect = HashingVectorizer(decode_error='ignore', n_features=2**21, preprocessor=None, tokenizer=tokenizer)
clf = SGDClassifier(loss='log', n_iter=1, n_jobs=-1)
stream = stream_docs('../../datasets/movie_data.csv')

for _ in range(45):
    X_train, y_train = get_minibatch(stream, 1000)
    if not  X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=[0,1])

X_test, y_test = get_minibatch(stream, 5000)
X_test=vect.transform(X_test)
print(("Accuracy: %.3f" % clf.score(X_test, y_test)))