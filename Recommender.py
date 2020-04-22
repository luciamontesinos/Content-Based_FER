import numpy as np
import pandas as pd

class Recommender:
    def __init__(self, df):
        self.df = df
        self.split = int(self.df.shape[0] * 0.2)
        self.input_user_ratings = None
        self.user_rated_genres = None
        self.df_copy = None

    def get_sample(self):
        self.df.drop(self.df.columns[[0, 1]], axis=1, inplace=True)
        return self.df.sample(n=self.split)

    def generate_user_rating(self, rating):
        # INVENTO INPUT USUARIO puntuación entre 0 y 1? o hay mas emociones-> mas puntuaciones
        #rating = np.random.randint(2, size=self.split)
        self.input_user_ratings = pd.DataFrame({'rating': rating})

    def make_recommendation(self, sample, rating):
        self.generate_user_rating(rating)
        #self.input_user_ratings = rating
        # Obtengo puntuacion a todas las canciones
        self.user_rated_genres = np.dot(self.input_user_ratings.T, sample)
        self.user_rated_genres = pd.DataFrame(data=self.user_rated_genres, index=['rating'], columns=self.df.columns)

        # Obtengo puntiacion por género
        self.user_rated_genres.loc['rating'].values[0] = 0  # Para no tener en cuenta el index
        self.df_copy = self.df.copy()
        self.df_copy['features'] = self.df_copy.apply(
            lambda x: [x['NUMERO'], x['POP'], x['ROCK'], x['INDIE / ALT'], x['HIP-HOP / RAP'], x['METAL'], x['PUNK'],
                       x['URBAN'], x['JAZZ'], x['ELECTRONICA'], x['FUNK']], axis=1)
        self.df_copy['score'] = self.df_copy.features.apply(
            lambda x: np.dot(x, self.user_rated_genres.loc['rating'].values))
        return self.df_copy.sort_values('score', ascending=False)
