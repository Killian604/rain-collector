"""
A shortcut script to cutting clips of movies
"""
import os
from moviepy.editor import AudioFileClip, VideoFileClip


def write_clip_audio_to_file(vidfile, outfile):
    vid = VideoFileClip(vidfile)
    vid.audio.write_audiofile(outfile)
    vid.close()


def cutclip(infile, outfile, timestart=0, timeend=-1, force=False):
    assert os.path.isfile(inputmoviefile), f'File not found: {inputmoviefile=}'
    assert inputmoviefile.endswith('.mp4'), f'Bad ext: {infile}'
    assert force or not os.path.isfile(outfile), f'File found: {outfile=}'
    clip = VideoFileClip(infile)
    if timeend < 0:
        timeend = clip.duration
    subclip = clip.subclip(timestart, timeend)

    subclip.write_videofile(outfile)

    clip.close()

    return


def cutaudio(infile, outfile, timestart=0, timeend=-1, force=False):
    """

    :param infile:
    :param outfile:
    :param timestart:
    :param timeend:
    :param force:
    :return:
    """
    assert os.path.isfile(inputmoviefile), f'File not found: {inputmoviefile=}'
    assert inputmoviefile.endswith('.mp3'), f'Bad ext: {infile}'
    assert force or not os.path.isfile(outfile), f'File found: {outfile=}'
    with AudioFileClip(infile) as clip:
        if timeend < 0:
            timeend = clip.duration
        subclip = clip.subclip(timestart, timeend)

        subclip.write_audiofile(outfile)
        # clip.close()
    return


# Cut video clip
if __name__ == '__main__' and False:
    inputmoviefile = f'/home/killfm/Downloads/jreAJ.mp4'
    outputmoviefile = f'/home/killfm/Downloads/testeroo1.mp4'
    timestart = 0
    timeend = 20
    cutaudio(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
    )

# Clipping an mp3
if __name__ == '__main__' and True:
    inputmoviefile = f'/home/killfm/Downloads/Comparing The Voices - Major Motoko Kusanagi (English).mp3'
    outputmoviefile = f'/home/killfm/Downloads/motoko1.mp3'
    timestart = 43
    timeend = 73
    cutaudio(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
    )

