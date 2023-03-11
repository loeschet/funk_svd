# funk_svd
A repository containing a recommendation engine for recommending movies to users based on the popular "FunkSVD" matrix factorization technique (made famous by [Simon Funk](https://sifter.org/simon/journal/20061211.html) in the [Netflix prize](https://en.wikipedia.org/wiki/Netflix_Prize) competition) as well as content and knowledge-based recommendations for users/movies that are not contained in the FunkSVD training data (e.g. because they are new users and/or movies)

## How to use this repository

The recommencation engine is implemented as the `Recommender` class which is contained in the `recommender.py` file. A more instructive introduction on how the recommendation engine works can be found in the `funksvd_rec.ipynb` jupyter notebook, where recommendations are made for user/movie data provided by [Udacity](https://www.udacity.com/) through the [data scientist nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025) course.

## Recommended software stack

A default modern python data analysis stack is necessary to run this project. In particular, users will need the following packages:

- numpy
- pandas
- matplotlib
- seaborn

