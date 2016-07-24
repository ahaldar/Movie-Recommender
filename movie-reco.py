# Original tutorial at http://online.cambridgecoding.com/notebooks/eWReNYcAfB/implementing-your-own-recommender-systems-in-python-2
#
# Python code using NumPy
# to create training and testing data matrices
# from given dataset of movie ratings by users
# and to predict ratings as a recommendation system
# and to compute corresponding error in predictions.
#

import numpy as np
import pandas as pd

header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('u.data', sep='\t', names=header)

n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
print 'Number of users = ' + str(n_users) + ' | Number of movies = ' + str(n_items)  

from sklearn import cross_validation as cv
train_data, test_data = cv.train_test_split(df, test_size=0.25)

#Create two user-item matrices, one for training and another for testing
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data:
                train_data_matrix[line[0]-1, line[1]-1] = line[2]

test_data_matrix = np.zeros((n_users, n_items))
for line in test_data:
                test_data_matrix[line[0]-1, line[1]-1] = line[2]

from sklearn.metrics.pairwise import pairwise_distances
user_similarity = pairwise_distances(train_data_matrix, metric='cosine')

def predict(ratings, similarity, type='user'):
                if type == 'user':
                                mean_user_rating = ratings.mean(axis=1)
                                ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
                                pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T   
                                return pred

user_prediction = predict(train_data_matrix, user_similarity, type='user')

from sklearn.metrics import mean_squared_error
from math import sqrt
def rmse(prediction, ground_truth):
                prediction = prediction[ground_truth.nonzero()].flatten() 
                ground_truth = ground_truth[ground_truth.nonzero()].flatten()
                return sqrt(mean_squared_error(prediction, ground_truth))
                
print 'User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix))

