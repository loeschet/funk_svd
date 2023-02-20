import numpy as np
import pandas as pd
from utils import (create_train_test, find_similar_movies, get_movie_names,
                   create_ranked_df, popular_recommendations)
from matplotlib import pyplot as plt
import seaborn as sns
from os.path import join
import json
import time
import sys

class Recommender():
    '''A recommender implementing FunkSVD and content-based rec. methods'''

    def __init__(self, mov_file, rev_file):
        '''The __init__method of the Recommender class
        
        Args:
            mov_file (str): string of path pointing to the csv file that
                contains the information about the movies
            rev_file (str): string of path pointing to the csv file that
                contains each user's movie reviews/ratings

        '''

        self._movs = pd.read_csv(mov_file)
        self._revs = pd.read_csv(rev_file)

        del self._movs['Unnamed: 0']
        del self._revs['Unnamed: 0']
        
        self._is_fitted = False

    def load_results(self, res_dir):
        
        # load fit settings/results JSON file
        with open(join(res_dir, "results.json"), "r") as jsonfile:
            res_dict = json.load(jsonfile)
        
        self.train_df, self.val_df = create_train_test(
            self._revs, res_dict["order_by"],  res_dict["train_size"],
            res_dict["test_size"])
        
        self.train_user_item = self.train_df.groupby(
            ['user_id', 'movie_id'])['rating'].max().unstack()
        
        self.u_mat = np.load(join(res_dir, "u_mat.npy"))
        self.i_mat = np.load(join(res_dir, "i_mat.npy"))
        
        self._is_fitted = True

    def fit(self, iterations = 100, latent_feats=12, train_size=8000,
            learning_rate=0.001, test_size=2000, order_by="date", seed=None):
        '''Fits the recommender to the dataset that is read when initializing
        
        Uses the FunkSVD method to fit recommender to the movie/rating datasets
        that were input in the initialization of this class.
        
        '''
        if (train_size + test_size) > self._revs.shape[0]:
            raise ValueError(("Error! Combined train and test set size "
                              "larger than currently available "
                              "amount of data!"))
        
        rng = np.random.default_rng(seed=seed)
        
        self.train_df, self.val_df = create_train_test(self._revs, order_by,
                                                       train_size, test_size)
        
        self.train_user_item = self.train_df.groupby(
            ['user_id', 'movie_id'])['rating'].max().unstack()
        
        train_user_item_mat = np.array(self.train_user_item)
        
        n_usrs = train_user_item_mat.shape[0]
        n_items = train_user_item_mat.shape[1]
        num_scores = np.count_nonzero(~np.isnan(train_user_item_mat))
        
        # randomly initialize matrices for predicted user and item scores
        u_mat = rng.uniform(size=(n_usrs, latent_feats))
        i_mat = rng.uniform(size=(latent_feats, n_items))

        # print iterations and MSE for each iteration
        print("Optimization")
        print("Iterations | Mean Squared Error")
        
        for i in range(iterations):
        # initialize sum of squared error
            sse = 0

            for row in range(n_usrs):
                for col in range(n_items):
                    if train_user_item_mat[row, col] > 0:
                        diff = (train_user_item_mat[row, col]
                                - np.dot(u_mat[row, :], i_mat[:, col]))

                        sse += diff**2

                        for k in range(latent_feats):
                            u_mat[row, k] += learning_rate*2*diff*i_mat[k, col]
                            i_mat[k, col] += learning_rate*2*diff*u_mat[row, k]

            print(f"{i+1} \t\t {sse / num_scores}")
        
        # set fitted matrices as class variables
        self.u_mat = u_mat
        self.i_mat = i_mat
        self._is_fitted = True
        
        print("Finished fitting.")

    def predict_rating(self, user_id, movie_id):
        '''
        makes predictions of a rating for a user on a movie-user combo
        '''
        
        if not self._is_fitted:
            raise ValueError(("Error! You are trying to predict the "
                              "ratings on a recommender that has not been "
                              "fitted yet! Please fit this recommender "
                              "and retry!"))
        
        user_ids = self.train_user_item.index
        movie_ids = self.train_user_item.columns
        
        if not (user_id in user_ids and movie_id in movie_ids):
            raise ValueError(("User or movie is not in training data and thus "
                              "no prediction can be made"))
        
        usr_row = np.where(user_ids == user_id)[0][0]
        movie_col = np.where(movie_ids == movie_id)[0][0]

        
        pred = np.dot(self.u_mat[usr_row], self.i_mat[:, movie_col])
        
        return pred

    # TODO: double-check this method
    def make_recs(self, id, id_type='movie', rec_num=5):
        
        '''
        INPUT:
        _id - either a user or movie id (int)
        _id_type - "movie" or "user" (str)

        rec_num - number of recommendations to return (int)

        
        OUTPUT:
        recs - (array) a list or numpy array of recommended movies like the 
                    given movie, or recs for a user_id given
        '''

        train_users = self.train_user_item.index
        train_movies = self.train_user_item.columns
        
        # if id type is movie, return list of recommended movies
        if id_type == "movie":
            recs = find_similar_movies(id)[:rec_num]
            recs = get_movie_names(recs, self._movs)
        elif id_type == "user":
            if id in train_users:
                # user is contained in training sample, so we can get movies
                # they havent seen yet and that have the highest rating
                idx = np.argwhere(train_users == id)[0][0]
                tmp_preds = np.dot(self.u_mat[idx], self.i_mat)
                # get top k
                sort_ind = np.argsort(tmp_preds)[(-1)*rec_num:][::-1]
                recs = train_movies[sort_ind]
                recs = get_movie_names(recs, self._movs)
            else:
                # user is NOT in training sample, we recommend movies based on
                # avg rating
                ranked_movs = create_ranked_df(self._movs, self.train_df)
                recs = popular_recommendations(id, self._revs, rec_num,
                                               ranked_movs)

        else:
            raise ValueError(("id_type must be 'movie' or 'user'! "
                              "Please retry!"))

        return recs

    def validation_comparison(self, plot=False):
        '''
        INPUT:
        val_df - the validation dataset created in the third cell above
        user_mat - U matrix in FunkSVD
        movie_mat - V matrix in FunkSVD
            
        OUTPUT:
        rmse - RMSE of how far off each value is from it's predicted value
        perc_rated - percent of predictions out of all possible that could be rated
        actual_v_pred - a 10 x 10 grid with counts for actual vs predicted values
        '''
        
        if not self._is_fitted:
            print(("Error! You are trying to evaluate the ratings of the "
                   "validation set on a recommender that has not been fitted "
                   "yet! Please fit this recommender and retry!"))

            return None
        
        val_users = np.array(self.val_df["user_id"])
        val_movies = np.array(self.val_df["movie_id"])
        val_ratings = np.array(self.val_df["rating"])
        
        sse = 0
        num_rated = 0
        preds = []
        vals =[]
        actual_v_pred = np.zeros((10,10))
        for idx in range(len(val_users)):
            try:
                pred = self.predict_rating(val_users[idx], val_movies[idx])
                sse += (val_ratings[idx]-pred)**2
                num_rated += 1
                preds.append(pred)
                vals.append(val_ratings[idx])
                if round(pred) > 10:
                    pred = 10
                elif round(pred) < 1:
                    pred = 1

                actual_v_pred[int(val_ratings[idx])-1, int(round(pred)-1)]+=1

            except ValueError:
                continue

        rmse = np.sqrt(sse/num_rated)
        perc_rated = num_rated/len(val_users)
        
        val_summary = {"rmse": rmse,
                    "perc_rated": perc_rated,
                    "preds": preds,
                    "vals": vals} 
        
        if plot:
            # plot confusion matrix as seaborn heatmap
            sns.heatmap(actual_v_pred)
            plt.xticks(np.arange(10), np.arange(1,11))
            plt.yticks(np.arange(10), np.arange(1,11))
            plt.xlabel("Predicted Values")
            plt.ylabel("Actual Values")
            plt.title("Actual vs. Predicted Values")
            plt.show()
            plt.close()
        
        return val_summary
