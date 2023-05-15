from youtube_crawler.downloader import Downloader
import pandas as pd
from tqdm import tqdm
from pytube import Playlist
import os

def crawl_from_youtube(url_path, data_path):
    url_df = pd.read_csv(url_path, names=["num_file", "url", "name"])
    downloader = Downloader()
    
    for index in url_df.index[2:]:
        dir_path = data_path + "/" + url_df["name"][index]
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        playlist_url = url_df["url"][index]
        playlist = Playlist(playlist_url)
        
        urls = []
        for url in playlist:
            urls.append(url)
        
        for url in urls[114:]:
            downloader.run(
                url=url,
                save_dir=dir_path,
                sampling_rate=22050
            )
            

if __name__ == "__main__":
    url_path = "datas/urls.txt"
    data_path = "datas/"
    crawl_from_youtube(
        data_path=data_path,
        url_path=url_path)