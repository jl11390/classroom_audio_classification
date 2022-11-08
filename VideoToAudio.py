import os
from os import listdir
from os.path import isfile, join
import json
from moviepy.editor import VideoFileClip
from pedalboard import Pedalboard, Distortion, PitchShift, Compressor, Bitcrush
from pedalboard.io import AudioFile


def convert_video_to_audio_moviepy(video_file, data_path, output_path, output_ext='wav', audio_fps=22050,
                                   overwrite=True):
    """
    Check whether the video has been extracted to audio
    Converts video to audio using MoviePy library that uses `ffmpeg` under the hood
    """
    filename, ext = os.path.splitext(video_file)
    if (os.path.exists(f"{output_path}/{filename}.{output_ext}") is False) or overwrite:
        clip = VideoFileClip(f'{data_path}/{video_file}', audio_fps=audio_fps)
        clip.audio.write_audiofile(f"{output_path}/{filename}.{output_ext}")
    else:
        print(f"{output_path}/{filename}.{output_ext} exists")


def augment_audio_data(audio_file, data_path, output_path,
                       aug_list=['compressor', 'distortion', 'pitchshift', 'bitcrush'], samplerate=22050,
                       overwrite=True):
    print(f'augmenting audio {audio_file}')
    aug_list_lower = [aug.lower() for aug in aug_list]
    filename, ext = os.path.splitext(audio_file)
    # init aug file name list
    file_aug_list = []
    # dict for all aug params
    aug_dict = {
        # change the params
        'compressor': Compressor(threshold_db=20, ratio=4),
        'distortion': Distortion(drive_db=25),
        'pitchshift': PitchShift(semitones=3),
        'bitcrush': Bitcrush(bit_depth=8)
    }
    # Read in a whole file, resampling to our desired sample rate:
    with AudioFile(f'{data_path}/{audio_file}').resampled_to(samplerate) as f:
        audio = f.read(f.frames)
    for aug in aug_list_lower:
        if (os.path.exists(f"{output_path}/{filename}_{aug}{ext}") is False) or overwrite:
            board = Pedalboard([aug_dict[aug]])
            effected = board(audio, samplerate)
            with AudioFile(f"{output_path}/{filename}_{aug}{ext}", 'w', samplerate, effected.shape[0]) as f:
                f.write(effected)
        else:
            print(f"{output_path}/{filename}_{aug}{ext} exists")
        # store the name in list for indexing
        file_aug_list.append(f"{filename}_{aug}{ext}")

    return {filename: file_aug_list}


if __name__ == "__main__":
    annot_path = 'data/COAS_2/Annotation'
    video_path = 'data/COAS_2/Videos'
    audio_path = 'data/COAS_2/Audios'
    audio_aug_path = 'data/COAS_2/Audios_augmented'
    audio_test_path = 'data/COAS_2/Audios_test'
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    if not os.path.exists(audio_aug_path):
        os.makedirs(audio_aug_path)
    if not os.path.exists(audio_test_path):
        os.makedirs(audio_test_path)
    # List all files under video folder
    videofiles = [f for f in listdir(video_path) if isfile(join(video_path, f)) and not f.startswith('.')]
    print(f'read in {len(videofiles)} videos')
    num_train = int(len(videofiles) * 0.9)
    # Extract audios from videos
    for i, file in enumerate(videofiles):
        print(f'extracted {i + 1} files and {len(videofiles) - i - 1} to go')
        if i <= num_train:
            convert_video_to_audio_moviepy(file, video_path, audio_path, output_ext="wav", overwrite=True)
        else:
            convert_video_to_audio_moviepy(file, video_path, audio_test_path, output_ext="wav", overwrite=True)

    # List all files under audio folder
    audiofiles = [f for f in listdir(audio_path) if f.endswith('wav')]
    print(f'{len(audiofiles)} audio files extracted for train')
    audiofiles_test = [f for f in listdir(audio_test_path) if f.endswith('wav')]
    print(f'{len(audiofiles_test)} audio files extracted for test')

    # start data augmentation on the training data
    file_aug_dict_all = {}
    for i, audio_file in enumerate(audiofiles):
        print(f'augmented {i + 1} files and {len(audiofiles) - i - 1} to go')
        file_aug_dict = augment_audio_data(audio_file, audio_path, audio_aug_path, aug_list=['compressor', 'distortion', 'pitchshift', 'bitcrush'], overwrite=True)
        file_aug_dict_all.update(file_aug_dict)

    jsonFile = open(f"{annot_path}/file_aug_dict.json", "w")
    jsonFile.write(json.dumps(file_aug_dict_all))
    jsonFile.close()
