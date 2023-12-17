import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier

import string


def remove_punctuation(text):
    return "".join([ch for ch in text if ch not in string.punctuation])


def remove_small_words(text):
    return " ".join([word for word in text.split() if len(word) > 3])


def stem_text(text):
    stem = PorterStemmer()
    return " ".join([stem.stem(word) for word in text.split()])


df = pd.read_csv("spam.csv")
df['Category'] = df['Category'].map(lambda x: 1 if x == 'spam' else 0)
df['Message'] = df['Message'].apply(remove_punctuation).apply(remove_small_words).apply(stem_text)

x = df['Message']
y = df['Category']

# num_words = len(' '.join(x).split())
# print(num_words)

# print(df.head(20))
x_train, x_test, y_train, y_test = train_test_split(x, y)

# Initialising TfidfVectoriser to remove english and to make lower case
tv = TfidfVectorizer(stop_words='english', lowercase=True)

# Extracting Features
x_train_features = tv.fit_transform(x_train)
X_test_features = tv.transform(x_test)

# print(len(tv.vocabulary_))
# print(tv.vocabulary_)

# Dictionary for Results
summary = {'Model': [], 'Accuracy': [],
           'TP': [], 'FP': [],
           'TN': [], 'FN': [],
           'FPR': [], 'FNR': [],
           'Precision': []}

# Creating Results for Multinomial Naive Bayes
MNB_model = MultinomialNB()
MNB_model.fit(x_train_features, y_train)
prediction = MNB_model.predict(X_test_features)
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, prediction).ravel()
false_positive_rate = false_positive / (false_positive + true_negative)
false_negative_rate = false_negative / (false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)

# Adding results to dictionary
summary['Model'].append('MultinomialNB')
summary['Accuracy'].append(accuracy)
summary['TP'].append(true_positive)
summary['FP'].append(false_positive)
summary['TN'].append(true_negative)
summary['FN'].append(false_negative)
summary['FPR'].append(false_positive_rate)
summary['FNR'].append(false_negative_rate)
summary['Precision'].append(precision)

# Creating Results for Support Vector Machine (SVM)
SVM_model = SVC()
SVM_model.fit(x_train_features, y_train)
svm_prediction = SVM_model.predict(X_test_features)
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, svm_prediction).ravel()
false_positive_rate = false_positive / (false_positive + true_negative)
false_negative_rate = false_negative / (false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)

# Adding results to dictionary
summary['Model'].append('SVM')
summary['Accuracy'].append(accuracy)
summary['TP'].append(true_positive)
summary['FP'].append(false_positive)
summary['TN'].append(true_negative)
summary['FN'].append(false_negative)
summary['FPR'].append(false_positive_rate)
summary['FNR'].append(false_negative_rate)
summary['Precision'].append(precision)

# Creating Results for Final Classifier
classifiers = [('mnb', MNB_model), ('svm', SVM_model)]
final_model = StackingClassifier(estimators=classifiers)

final_model.fit(x_train_features, y_train)
prediction = final_model.predict(X_test_features)
true_negative, false_positive, false_negative, true_positive = confusion_matrix(y_test, prediction).ravel()
false_positive_rate = false_positive / (false_positive + true_negative)
false_negative_rate = false_negative / (false_negative + true_positive)
precision = true_positive / (true_positive + false_positive)
accuracy = (true_positive + true_negative) / (true_negative + false_positive + false_negative + true_positive)

# Adding results to dictionary
summary['Model'].append('Stack')
summary['Accuracy'].append(accuracy)
summary['TP'].append(true_positive)
summary['FP'].append(false_positive)
summary['TN'].append(true_negative)
summary['FN'].append(false_negative)
summary['FPR'].append(false_positive_rate)
summary['FNR'].append(false_negative_rate)
summary['Precision'].append(precision)

result = pd.DataFrame(summary)
print(result)
