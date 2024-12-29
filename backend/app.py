import os
import uvicorn
from moviepy.video.io.VideoFileClip import VideoFileClip
from uvicorn.config import LOGGING_CONFIG
from moviepy.editor import AudioFileClip, VideoFileClip
import os
# from file_monitor import WatchdogThread, UpdateThread
from .model_server import fastapiapp
# from settings import *
from pydub import AudioSegment



def mp3towav(inputfp, outputfp, force: bool = False):
    """"""
    if not os.path.isfile(inputfp):
        raise FileNotFoundError(f'File not found: {inputfp=}')
    if os.path.isfile(outputfp) and not force:
        raise FileExistsError(f'File already exists: {outputfp=}')

    sound = AudioSegment.from_mp3(inputfp)
    sound.export(outputfp, format="wav")



def write_clip_audio_to_file(vidfile, outfile):
    with VideoFileClip(vidfile) as vid:
        vid.audio.write_audiofile(outfile)
    print(f'Audio file saved at: {outfile}')


def mp4tomp3(input_vid_fp, output_mp3_fp, force: bool = False):
    """Convert mp4 to mp3 to extract audio"""
    if not force and os.path.isfile(output_mp3_fp):
        raise FileExistsError(f'Output already exists: {output_mp3_fp}')
    video = VideoFileClip(input_vid_fp)
    video.write_audiofile(output_mp3_fp)
    print(f'mp3 saved to: {output_mp3_fp}')


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
    return


def cutclip(infile, outfile, timestart=0, timeend=-1, force=False):
    if not os.path.isfile(inputmoviefile): raise FileNotFoundError(f'File not found: {inputmoviefile=}')
    if not inputmoviefile.endswith('.mp4'): raise ValueError(f'Bad ext: {infile}. Expects to be an mp4 input')
    if not os.path.isfile(inputmoviefile): raise FileNotFoundError(f'File not found: {inputmoviefile=}')
    if not force and os.path.isfile(outfile): raise FileExistsError(f'Output file found: {outfile=}')
    with VideoFileClip(infile) as clip:
        if timeend < 0:
            timeend = clip.duration
        with clip.subclip(timestart, timeend) as subclip:
            subclip.write_videofile(outfile)


# Cut video clip
if __name__ == '__main__' and True:
    inputmoviefile = f'/home/killfm/Videos/jreAJ.mp4'
    outputmoviefile = f'/home/killfm/Videos/deleteme102.mp4'
    timestart = 32*60+30
    timeend = 32*60+30+5
    cutclip(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
    )


# Clipping an mp3
if __name__ == '__main__' and False:
    inputmoviefile = f'/home/killfm/Downloads/Comparing The Voices - Major Motoko Kusanagi (English).mp3'
    inputmoviefile = f'/home/killfm/Videos/motoko1.mp3'
    outputmoviefile = f'/home/killfm/Videos/motoko1_cut1.mp3'
    timestart = 10
    timeend = 20
    cutaudio(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
    )


if __name__ == "__main__" and False:
    # watchdog_thread = WatchdogThread(CHAT_PATH, NOTE_PATH)
    # watchdog_thread.start()
    # update_thread = UpdateThread(server_state)
    # update_thread.start()

    LOGGING_CONFIG["formatters"]["access"]["fmt"] = "%(asctime)s %(levelprefix)s %(message)s"
    LOGGING_CONFIG["formatters"]["access"]["datefmt"] = "%Y-%m-%d %H:%M:%S"
    uvicorn.run(
        fastapiapp,
        host=os.getenv("HOST", "localhost"),
        port=os.getenv("PORT", 5000),
    )
