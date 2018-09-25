#% matplotlib
#inline
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas
import sklearn
import cPickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.learning_curve import learning_curve

## loading data using Pandas

messages = pandas.read_csv(
    'sms_spam.csv',
    names = ["label", "message"])
print(messages)

## To get pivot of dataset

messages.groupby('label').describe()

## adding extra column to get the length of the text

messages['length'] = messages['message'].map(lambda text: len(text))
print(messages.head())

## plotting

messages.length.plot(bins=20, kind='hist')

messages.length.describe()

## to check the longest messages

print(list(messages.message[messages.length > 900]))

## checking out the difference between ham and spam

messages.hist(column='length', by='label', bins=50)


## Data Preprocessing

## we'll use the bag-of-words approach, where each unique word in a text will be represented by one number.

## splitting it into tokens

def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return TextBlob(message).words


## original text

messages.message.head()

## now text after tokenized

messages.message.head().apply(split_into_tokens)

## Part of speech tag (POS)

TextBlob("Hello world, how is it going?").tags  # list of (word, POS) pairs


## lemmatizing --- normalize words into their base form

def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma
    return [word.lemma for word in words]


messages.message.head().apply(split_into_lemmas)

## Data to Vectors

# Now we'll convert each message, represented as a list of tokens (lemmas) above,
# into a vector that machine learning models can understand.

# Doing that requires essentially three steps, in the bag-of-words model:

# 1. counting how many times does a word occur in each message (term frequency)
# 2. weighting the counts, so that frequent tokens get lower weight (inverse document frequency)
# 3. normalizing the vectors to unit length, to abstract from the original text length (L2 norm)

# Each vector has as many dimensions as there are unique words in the SMS corpus:

bow_transformer = CountVectorizer(analyzer = split_into_lemmas).fit(messages['message'])
print(len(bow_transformer.vocabulary_))

# Here we used `scikit-learn` (`sklearn`), a powerful Python library for teaching machine learning.
# It contains a multitude of various methods and options.

# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new `bow_transformer`:

## Feature Engineering

message4 = messages['message'][3]
print(message4)

bow4 = bow_transformer.transform([message4])
print(bow4)
print(bow4.shape)

# So, nine unique words in message nr. 4, two of them appear twice, the rest only once.
# lets check what are these words the appear twice?

print(bow_transformer.get_feature_names()[6736])
print(bow_transformer.get_feature_names()[8013])

messages_bow = bow_transformer.transform(messages['message'])
print('sparse matrix shape:', messages_bow.shape)
print('number of non-zeros:', messages_bow.nnz)
print('sparsity: %.2f%%' % (100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])))

# And finally, after the counting, the term weighting and normalization
# can be done with [TF-IDF] using scikit-learn's `TfidfTransformer`:

tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)

# To check what is the IDF (inverse document frequency) of the word `"u"`? Of word `"university"`?

print
tfidf_transformer.idf_[bow_transformer.vocabulary_['u']]
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])

# To transform the entire bag-of-words corpus into TF-IDF corpus at once:

messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

## Training a model
# We'll be using scikit-learn here, choosing the [Naive Bayes](http://en.wikipedia.org/wiki/Naive_Bayes_classifier)
# classifier to start with:

#% time
spam_detector = MultinomialNB().fit(messages_tfidf, messages['label'])




# Let's try classifying our single random message:
print('predicted:', spam_detector.predict(tfidf4)[0])
print('expected:', messages.label[3])

all_predictions = spam_detector.predict(messages_tfidf)
print(all_predictions)

## Calculating accuracy and confusion matrix on Training data which will definitely give good accuracy

print('accuracy', accuracy_score(messages['label'], all_predictions))
print('confusion matrix\n', confusion_matrix(messages['label'], all_predictions))
print('(row=expected, col=predicted)')

plt.matshow(confusion_matrix(messages['label'], all_predictions), cmap=plt.cm.binary, interpolation='nearest')
plt.title('confusion matrix')
plt.colorbar()
plt.ylabel('expected label')
plt.xlabel('predicted label')

print(classification_report(messages['label'], all_predictions))

## splitting the data into training and testing

msg_train, msg_test, label_train, label_test = \
    train_test_split(messages['message'], messages['label'], test_size=0.2)

print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

## Cross Validation

# A common practice is to partition the training set again, into smaller subsets; for example, 5 equally sized subsets.
# Then we train the model on four parts, and compute accuracy on the last part (called "validation set").
# Repeated five times (taking different part for evaluation each time), we get a sense of model "stability".
# If the model gives wildly different scores for different subsets, it's a sign something is wrong (bad data, or bad model variance).
# Go back, analyze errors, re-check input data for garbage, re-check data cleaning.

scores = cross_val_score(pipeline,  # steps to convert raw messages into models
                         msg_train,  # training data
                         label_train,  # training labels
                         cv=10,  # split data randomly into 10 parts: 9 for training, 1 for scoring
                         scoring='accuracy',  # which scoring metric?
                         n_jobs=-1,  # -1 = use all cores = faster
                         )
print(scores)
print(scores.mean(), scores.std())


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

#% time
plot_learning_curve(pipeline, "accuracy vs. training set size", msg_train, label_train, cv=5)

# At this point, we have two options:

# 1. use more training data, to overcome low model complexity
# 2. use a more complex (lower bias) model to start with, to get more out of the existing data

params = {
    'tfidf__use_idf': (True, False),
    'bow__analyzer': (split_into_lemmas, split_into_tokens),
}

grid = GridSearchCV(
    pipeline,  # pipeline from above
    params,  # parameters to tune via cross validation
    refit=True,  # fit using all available data at the end, on the best found param combination
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

#% time
nb_detector = grid.fit(msg_train, label_train)
print(nb_detector.grid_scores_)

print(nb_detector.predict_proba(["Hi mom, how are you?"])[0])
print(nb_detector.predict_proba(["WINNER! Credit for free!"])[0])

print(nb_detector.predict(["Hi mom, how are you?"])[0])
print(nb_detector.predict(["WINNER! Credit for free!"])[0])

# And overall scores on the test set, the one we haven't used at all during training

predictions = nb_detector.predict(msg_test)
print(confusion_matrix(label_test, predictions))
print(classification_report(label_test, predictions))

#############################################################################################################

################## SVM #########################################

pipeline_svm = Pipeline([
    ('bow', CountVectorizer(analyzer=split_into_lemmas)),
    ('tfidf', TfidfTransformer()),
    ('classifier', SVC()),  # <== change here
])

# pipeline parameters to automatically explore and tune
param_svm = [
    {'classifier__C': [1, 10, 100, 1000], 'classifier__kernel': ['linear']},
    {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf']},
]

grid_svm = GridSearchCV(
    pipeline_svm,  # pipeline from above
    param_grid=param_svm,  # parameters to tune via cross validation
    refit=True,  # fit using all data, on the best detected classifier
    n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
    scoring='accuracy',  # what score are we optimizing?
    cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
)

#%time
svm_detector = grid_svm.fit(msg_train, label_train)  # find the best combination from param_svm
print(svm_detector.grid_scores_)

print(svm_detector.predict(["Hi mom, how are you?"])[0])
print(svm_detector.predict(["WINNER! Credit for free!"])[0])

print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
print(classification_report(label_test, svm_detector.predict(msg_test)))

print(confusion_matrix(label_test, svm_detector.predict(msg_test)))
print(classification_report(label_test, svm_detector.predict(msg_test)))

## Productionalizing a predictor

# store the spam detector to disk after training
with open('sms_spam_detector.pkl', 'wb') as fout:
    cPickle.dump(svm_detector, fout)

# ...and load it back, whenever needed, possibly on a different machine
svm_detector_reloaded = cPickle.load(open('sms_spam_detector.pkl'))

print('before:', svm_detector.predict([message4])[0])
print('after:', svm_detector_reloaded.predict([message4])[0])