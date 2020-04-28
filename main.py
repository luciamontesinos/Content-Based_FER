import pandas as pd
import recommender
from emotion_detector import  EmotionDetector

if __name__ == '__main__':
    df = pd.read_excel("./resources/song_genre_dataset.xlsx", sheet_name=0)
    playlist = df.copy()
    playlist.drop(df.iloc[:, 3:], inplace=True, axis=1)
    recommender = recommender.Recommender(df)
    sample = recommender.get_sample()
    songs = sample['NUMERO'].values
    emotion_detector = EmotionDetector()
    detected = emotion_detector.start_detection(songs)
    print(detected)
    recommendation = recommender.make_recommendation(sample, detected)
    print(playlist.loc[playlist['NUMERO'].isin(recommendation['NUMERO'][:5])])
