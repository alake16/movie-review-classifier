import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
	return array

X_train, X_dev, X_test, y_train, y_dev, y_test = parse_dataset()
feature_vector = create_feature_vector(X_train["review"], 5, 5000)
array = map_reviews_to_array(X_train["review"], feature_vector)
