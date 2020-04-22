##import vlc
##
##player = vlc.MediaPlayer("./nil-moliner-mi-religion-videoclip-oficial.mp3")
##player.play()
import threading

import vlc
import time

#
#def play_song(song):
#    instance = vlc.Instance()
#    player = instance.media_player_new()
#    media = instance.media_new(song)
#    player.set_media(media)
#
#    player.play()
#
#    time.sleep(5)  # Dar un tiempo prudencial para que la reproducción inicie.
#    while player.is_playing():
#       time.sleep(1)

import vlc
import time
import emotion_detector
from emotion_detector import emotion_determinator
def player():
    Instance = vlc.Instance()
    playlist = ['nil-moliner-mi-religion-videoclip-oficial.mp3', 'seguridad-social-quiero-tener-tu-presencia.mp3', 'bloc-party-banquet.mp3']
    for song in playlist:
        song = "playlist/" + song
        print('Sampling for 15 seconds')
        player = Instance.media_player_new()
        Media = Instance.media_new(song)
        Media_list = Instance.media_list_new([song])
        Media.get_mrl()
        player.set_media(Media)
        list_player = Instance.media_list_player_new()
        list_player.set_media_list(Media_list)
        if list_player.play() == -1:
            print ("Error playing playlist")
        time.sleep(15)
        list_player.stop()
        song_ended = False






