import os

from backend.app import mp3towav

inputfile = '/home/killfm/Videos/motoko1.mp3'
outputfile = '/home/killfm/Videos/motoko1.wav'

assert not os.path.isfile(outputfile), f'Output already exists: {outputfile=}'
mp3towav(inputfile, outputfile)
