import threading
import time
import vlc
from tensorflow import keras
import numpy as np
import cv2


class EmotionDetector:
    def __init__(self):
        self.emotion = ['Sad', 'Happy', 'Neutral']
        self.model = keras.models.load_model('./resources/my_model.h5')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.cam = cv2.VideoCapture(0)
        self.face_cas = cv2.CascadeClassifier('./resources/cascades/haarcascade_frontalface_default.xml')
        self.detected_emotions = []
        self.emotion_list = []
        self.song_ended = False
        self.playlist_ended = False
        self.counter = 0
        self.punctuated_emotions = []

    def emotion_determinator(self):
        while True:
            ret, frame = self.cam.read()
            frame = cv2.flip(frame, 1)
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # gray = cv2.flip(gray,1)
                faces = self.face_cas.detectMultiScale(gray, 1.3, 5)
                self.emotion_seeker(faces, gray, frame)
                if cv2.waitKey(1) == 27 or self.playlist_ended:
                    break

            else:
                print('Error')

        self.cam.release()
        cv2.destroyAllWindows()
        self.punctuated_emotions = self.emotion_converter(self.emotion_list)
        print(len(self.punctuated_emotions))


    def emotion_seeker(self, faces, gray, frame):
        for (x, y, w, h) in faces:
            face_component = gray[y:y + h, x:x + w]
            fc = cv2.resize(face_component, (48, 48))
            inp = np.reshape(fc, (1, 48, 48, 1)).astype(np.float32)
            inp = inp / 255.
            prediction = self.model.predict_proba(inp)
            em = self.emotion[np.argmax(prediction)]
            self.detected_emotions.append(em)
            score = np.max(prediction)
            cv2.putText(frame, em + "  " + str(score * 100) + '%', (x, y), self.font, 1, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            if self.song_ended and self.counter == 0:
                final_emotion = max(set(self.detected_emotions), key=self.detected_emotions.count)
                print("Song generated this emotion:", final_emotion)
                self.emotion_list.append(final_emotion)
                self.detected_emotions.clear()
                self.counter = 1
        cv2.imshow("image", frame)

    def emotion_converter(self, emos):
        punctuation = []
        for i in range(len(emos)):
            if emos[i] == "Happy":
                punctuation.append(5)
            elif emos[i] == "Neutral":
                punctuation.append(2)
            else:
                punctuation.append(0)
        return punctuation

    def player(self, playlist):
        Instance = vlc.Instance()
        print(playlist)
        time.sleep(1)
        for song in playlist:
            self.counter = 0
            self.song_ended = False
            song = "./resources/playlist/" + str(song) + ".mp3"
            print('Sampling...')
            player = Instance.media_player_new()
            Media = Instance.media_new(song)
            Media.add_option('start-time=40.00')
            player.set_media(Media)
            if player.play() == -1:
                print("Error playing playlist")
            time.sleep(5)#15
            player.stop()
            self.song_ended = True
            time.sleep(1)
            self.song_ended = False
        self.playlist_ended = True

    def start_detection(self, playlist):
        emotion_t = threading.Thread(target=self.emotion_determinator)
        music_t = threading.Thread(target=self.player, args=[playlist])

        emotion_t.start()
        music_t.start()

        emotion_t.join()
        music_t.join()
        return self.punctuated_emotions



