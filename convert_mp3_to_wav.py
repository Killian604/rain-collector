from moviepy.editor import AudioFileClip
import os


inputfile = '/home/killfm/Videos/motoko1.mp3'
outputfile = '/home/killfm/Videos/motoko1.wav'


def mp3_to_wav(mp3_file, outpath):
    clip = AudioFileClip(mp3_file)
    clip.write_audiofile(outpath)

assert not os.path.isfile(outputfile), f'Output already exists: {outputfile=}'
mp3_to_wav(inputfile, outputfile)
