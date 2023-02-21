import pandas as pd
import numpy as np


def create_train_test(reviews, order_by, training_size, testing_size):
    '''Split a data into a training and test set based on a specified column.

    Args:
        reviews (pandas.DataFrame): Dataframe to split into train and test.
        order_by (str): Column name to sort by.
        training_size (int): Number of rows in the training set.
        testing_size (int): Number of rows in the test set.

    Returns:
        pandas.DataFrame: Dataframe of the training set.
        pandas.DataFrame: Dataframe of the test set.
    '''
    reviews_new = reviews.sort_values(order_by)
    training_df = reviews_new.head(training_size)
    validation_df = reviews_new.iloc[training_size:training_size+testing_size]

    return training_df, validation_df


def find_similar_movies(movie_id, movies):
    '''Find the most similar movies based on a given movie_id.

    Args:
        movie_id (int): A movie_id.
        movies (pandas.DataFrame): A dataframe created from the movies.csv
            file.

    Returns:
        numpy.ndarray: An array of the most similar movies by title.
    '''
    # find the row of each movie id
    movie_idx = np.where(movies['movie_id'] == movie_id)[0][0]

    # Subset so movie_content is only using the dummy variables for each genre
    # and the 3 century based year dummy columns
    movie_content = np.array(movies.iloc[:, 4:])

    # Take the dot product to obtain a movie x movie matrix of similarities
    dot_prod_movies = movie_content.dot(np.transpose(movie_content))

    # find the most similar movie indices - to start I said they need to be
    # the same for all content
    similar_idxs = np.where(
        dot_prod_movies[movie_idx] == np.max(dot_prod_movies[movie_idx]))[0]

    # pull the movie titles based on the indices
    similar_movies = np.array(movies.iloc[similar_idxs, ]['movie'])

    return similar_movies


def get_movie_names(movie_ids, movies):
    '''Find the names of movies associated with a given list of movie_ids.

    Args:
        movie_ids (list): A list of movie_ids.

    Returns:
        list: A list of movie names associated with the movie_ids.
    '''
    movie_lst = list(movies[movies['movie_id'].isin(movie_ids)]['movie'])

    return movie_lst


def create_ranked_df(movies, reviews):
    '''Rank movies based on ag. rating, number of reviews, recency of reviews.

    Args:
        movies (pandas.DataFrame): The movies dataframe.
        reviews (pandas.DataFrame): The reviews dataframe.

    Returns:
        pandas.DataFrame: A dataframe with movies that are sorted by highest
        average rating, more reviews, then time, and must have more than
        4 ratings.
    '''

    # Pull the average ratings and number of ratings for each movie
    movie_ratings = reviews.groupby('movie_id')['rating']
    avg_ratings = movie_ratings.mean()
    num_ratings = movie_ratings.count()
    last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
    last_rating.columns = ['last_rating']

    # Add Dates
    rating_count_df = pd.DataFrame({'avg_rating': avg_ratings,
                                    'num_ratings': num_ratings})

    rating_count_df = rating_count_df.join(last_rating)

    # merge with the movies dataset
    movie_recs = movies.set_index('movie_id').join(rating_count_df)

    # sort by top avg rating and number of ratings
    ranked_movies = movie_recs.sort_values(
        ['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

    # for edge cases - subset the movie list to those with only 5 or
    # more reviews
    ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

    return ranked_movies


def popular_recommendations(user_id, revs, n_top, ranked_movies):
    '''Generate top movie recommendations for a user based on their reviews.

    Args:
        user_id (str): The user_id of the individual you are making
            recommendations for.
        revs (pandas.DataFrame): DataFrame containing user reviews.
        n_top (int): The number of recommendations to return.
        ranked_movies (pandas.DataFrame): A dataframe of already ranked movies
            based on average rating, count, and time.

    Returns:
        list: A list of the n_top recommended movies by movie title in order
        best to worst.
    '''

    # remove movies already seen by user and recommend top ranked movies
    seen_movies = revs[revs["user_id"] == user_id]["movie_id"].values
    ranked_movies = ranked_movies[~ranked_movies.index.isin(seen_movies)]
    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies
