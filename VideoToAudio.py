import os
from os import listdir
from os.path import isfile, join
from moviepy.editor import VideoFileClip


def convert_video_to_audio_moviepy(video_file, data_path, output_path, output_ext="wav", audio_fps=22050):
    """
    Check whether the video has been extracted to audio
    Converts video to audio using MoviePy library that uses `ffmpeg` under the hood
    """
    filename, ext = os.path.splitext(video_file)
    if os.path.exists(f"{output_path}/{filename}.{output_ext}") is False:
        clip = VideoFileClip(f'{data_path}/{video_file}', audio_fps=audio_fps)
        clip.audio.write_audiofile(f"{output_path}/{filename}.{output_ext}")
    else:
        print(f"{output_path}/{filename}.{output_ext} exists")


if __name__ == "__main__":
    data_path = 'data/COAS/Videos'
    output_path = 'data/COAS/Audios'
    output_test_path = 'data/COAS/Audios_test'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_test_path):
        os.makedirs(output_test_path)
    # List all files under video folder
    videofiles = [f for f in listdir(data_path) if isfile(join(data_path, f)) and not f.startswith('.')]
    num_train = int(len(videofiles) * 0.9)
    # Extract audios from videos
    for i, file in enumerate(videofiles):
        if i <= num_train:
            convert_video_to_audio_moviepy(file, data_path, output_path, output_ext="wav")
        else:
            convert_video_to_audio_moviepy(file, data_path, output_test_path, output_ext="wav")

    # List all files under audio folder
    audiofiles = [f for f in listdir(output_path) if f.endswith('wav')]
    print(f'{len(audiofiles)} audio files extracted for train')
    audiofiles_test = [f for f in listdir(output_test_path) if f.endswith('wav')]
    print(f'{len(audiofiles_test)} audio files extracted for train')
