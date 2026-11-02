# converts the videos to mp3
import whisper
import os 
import subprocess
files = os.listdir("videos")
for file in files :
   print(file)
   tutorial_number= file.split("_")[0].split("-")[1]###to get lecture nu,bers
   file_name=file.split("_")[1]
   print(tutorial_number,file_name)
   subprocess.run(["ffmpeg", "-i", f"videos/{file}", f"audios/{tutorial_number}_{file_name}.mp3"])