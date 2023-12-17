import string
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import StackingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from GUI import run_gui


def remove_punctuation(text):
    return "".join([ch for ch in text if ch not in string.punctuation])


def remove_small_words(text):
    return " ".join([word for word in text.split() if len(word) > 3])


def stem_text(text):
    stem = PorterStemmer()
    return " ".join([stem.stem(word) for word in text.split()])


dataset = pd.read_csv("spam.csv")
# print(ds)
dataset['Category'] = dataset['Category'].map(lambda x: 1 if x == 'spam' else 0)
dataset['Message'] = dataset['Message'].apply(remove_punctuation).apply(remove_small_words).apply(stem_text)

y = dataset['Category']
x = dataset['Message']

# print(y)
# print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y)
# print(x_train)
# print(x_test)

tv = TfidfVectorizer(stop_words='english', lowercase=True)

x_train_features = tv.fit_transform(x_train)
# print(len(tv.vocabulary_))
# print(tv.vocabulary_)
estimators = [('mnb', MultinomialNB()), ('svm', SVC())]
final_model = StackingClassifier(estimators=estimators)
final_model.fit(x_train_features, y_train)

model = {'Stack': final_model}

# Run GUI
run_gui(model, tv)
