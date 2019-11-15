import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from random import seed
from random import randrange
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def fetch_reviews(filename, sentiment):
	reviews = []
	words = []
	f = open(filename, "r")
	lines = f.readlines()
	f.close()
	for line in lines:
		reviews.append(line)
	df = pd.DataFrame(reviews, columns =["review"])
	df["sentiment"] = 1 if sentiment == "pos" else 0
	return df

def parse_dataset():
	neg_reviews = fetch_reviews("./data/rt-polarity.neg", "neg")
	pos_reviews = fetch_reviews("./data/rt-polarity.pos", "pos")
	reviews = pd.concat([neg_reviews, pos_reviews], axis=0)
	# Split Train and Test sets
	X_train, X_test, y_train, y_test = train_test_split(reviews.drop("sentiment", 1), reviews["sentiment"], test_size = .15, random_state=0)
	# Split Train and Dev sets
	X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size = .15, random_state=0)
	return X_train, X_dev, X_test, y_train, y_dev, y_test

def create_feature_vector(reviews, least_num_occurrences, max_num_occurrences):
	word_occurences = {}
	for review in reviews:
		for word in review.split(" "):
			if word in word_occurences:
				word_occurences[word] += 1
			else:
			    word_occurences[word] = 1
	i = 0
	feature_vector = {}
	for word in word_occurences:
		if word_occurences[word] > least_num_occurrences and word_occurences[word] < max_num_occurrences:
			feature_vector[word] = i
			i += 1
	return feature_vector

def map_reviews_to_array(reviews, feature_vector):
	array = np.empty([len(reviews), len(feature_vector)])
	i = 0
	for review in reviews:
		word_list = np.zeros(len(feature_vector))
		for word in review.split(" "):
			if word in feature_vector:
				word_list[feature_vector[word]] += 1
		array[i] = word_list
		i += 1
	columns = {}
	for i in range(0, len(array[0])):
		columns[str(i)] = array[:, i]
	df = pd.DataFrame(columns)
	return df

def cross_validation_split(data, folds=3):
	seed(1)
	dataset_split = []
	dataset_copy = list(data)
	fold_size = len(data) // folds
	# print(fold_size)
	for i in range(folds):
		fold = []
		while len(fold) < fold_size:
			length = len(dataset_copy)
			# print(len(fold))
			index = randrange(length)
			dataset_copy.pop(index)
			fold.append(index)
		dataset_split.append(np.array(fold))
	return np.array(dataset_split)

def n_fold_cross_validation(X, y, model, split):
	scores = []
	i = 0
	for test_index in split:
		train_list = []
		print("Split {}/{}".format(i + 1, len(split)))
		for j in range(0, len(split)):
			if (i != j):
				train_list.append(split[j])
		i += 1
		train_index = np.concatenate(train_list, axis=None)
		X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
		model.fit(X_train, y_train.values.ravel())
		scores.append(model.score(X_test, y_test))
	return np.mean(scores)

def grid_serach(X_train, y_train, split):
	penalty = ['l1', 'l2']
	c = [1.0, 0.1, 0.01, 0.001, 0.0001]
	param_grid = dict(penalty=penalty, c=c)
	best_score = 0
	best_model = LogisticRegression()
	for penalty_val in penalty:
		for c_val in c:
			model = LogisticRegression(penalty=penalty_val, C=c_val)
			score = n_fold_cross_validation(X_train, y_train, model, split)
			print("Cross validation score using {} penalty and C val of {}: {}".format(penalty_val, c_val, score))
			if (score > best_score):
				best_model = model
	return best_model

def predict_test_data(model, X_test, y_test):
	preds = model.predict(X_test)
	acc_score = accuracy_score(y_test, preds)
	print("Model accuracy score on test data: {}".format(acc_score))
	prec_score = precision_score(y_test, preds)
	print("Model precision score on test data: {}".format(prec_score))
	rec_score = recall_score(y_test, preds)
	print("Model recall score on test data: {}".format(rec_score))
	f1_sco = f1_score(y_test, preds)
	print("Model f1 score on test data: {}".format(f1_sco))

X_train, X_dev, X_test, y_train, y_dev, y_test = parse_dataset()
feature_vector = create_feature_vector(X_train["review"], 5, 5000)
X_train = map_reviews_to_array(X_train, feature_vector)
X_test = map_reviews_to_array(X_test, feature_vector)
cols = {"sentiment": y_train}
y_train = pd.DataFrame(cols)
X_train.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
df = pd.concat([X_train, y_train], axis=1)
split = cross_validation_split(df.values.tolist(), 10)
model = grid_serach(X_train, y_train, split)
predict_test_data(model, X_test, y_test)
