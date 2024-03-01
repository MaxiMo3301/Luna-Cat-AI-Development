from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2 import service_account
import webbrowser

def play_music(song):

    # Set up API client
    DEVELOPER_KEY = "AIzaSyAJUHcPmM7quJNaW_qotSii52V83WY6zn4"
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=DEVELOPER_KEY)

    # Search for videos
    search_response = youtube.search().list(
        q = song,
        part = "id",
        type = "video",
        fields = "items/id"
    ).execute()

    # Get the first video ID
    video_id = search_response["items"][0]["id"]["videoId"]

    # Play the video in a web browser
    webbrowser.open(f"https://www.youtube.com/watch?v={video_id}") 



if __name__ == "__main__":

    music_name = "Lucid Dream"

    play_music(music_name)