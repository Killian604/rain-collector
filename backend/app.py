import os
import uvicorn
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from uvicorn.config import LOGGING_CONFIG
from moviepy.editor import AudioFileClip, VideoFileClip
import os
# from file_monitor import WatchdogThread, UpdateThread
from backend.model_server import fastapiapp
# from settings import *
from pydub import AudioSegment


# AV

def mp3towav(inputfp, outputfp, force: bool = False):
    """"""
    if not os.path.isfile(inputfp):
        raise FileNotFoundError(f'File not found: {inputfp=}')
    if os.path.isfile(outputfp) and not force:
        raise FileExistsError(f'File already exists: {outputfp=}')

    sound = AudioSegment.from_mp3(inputfp)
    sound.export(outputfp, format="wav")


def write_clip_audio_to_file(vidfile, outfile, force: bool = False):
    # TODO: Force
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
    if not os.path.isfile(inputmoviefile):
        raise FileNotFoundError(f'File not found: {inputmoviefile=}')
    # if not inputmoviefile.endswith('.mp4'): raise ValueError(f'Bad ext: {infile}. Expects to be an mp4 input')
    if not os.path.isfile(inputmoviefile):
        raise FileNotFoundError(f'File not found: {inputmoviefile=}')
    if not force and os.path.isfile(outfile):
        raise FileExistsError(f'Output file found: {outfile=}')
    with VideoFileClip(infile) as clip:
        if timeend < 0:
            timeend = clip.duration
        with clip.subclip(timestart, timeend) as subclip:
            subclip.write_videofile(outfile)


# Everything else
def asr_whisper(mp3path) -> str:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
    import time
    import torch
    import os
    os.environ['TORCHDYNAMO_VERBOSE'] = '1'

    # mp3path = '/home/killfm/Downloads/cbtlessons178b43a886a10a7c89cecabaebd6eb61750c5cd74.m4a'
    # mp3path = '/home/killfm/Downloads/jreAJ.mp3'
    # mp3path = f'/home/killfm/Videos/motoko1_cut1.mp3'

    assert os.path.isfile(mp3path), f'FNF: {mp3path=}'

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f'{device=}')
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        # safetensors=True,  #
    )
    model.to(device)

    # # Enable static cache and compile the forward pass
    # model.generation_config.cache_implementation = "static"
    # model.generation_config.max_new_tokens = 256
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)

    processor = AutoProcessor.from_pretrained(model_id)
    time_start = time.perf_counter()

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True
    )

    result = pipe(mp3path)
    time_end = time.perf_counter()
    elapsed_secs = time_end - time_start
    # print(f'{result=}')
    text = result["text"]
    print(f'{text}')
    print(f'Time to process transcription: {elapsed_secs:.2f} seconds')
    return text
    # breakpoint()


# Deprecated
def mp3_to_wav(mp3_file, outpath):
    """Deprecated due to duplication but works

    :param mp3_file:
    :param outpath:
    :return:
    """
    clip = AudioFileClip(mp3_file)
    clip.write_audiofile(outpath)


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

# Cut video clip
if __name__ == '__main__' and False:
    inputmoviefile = f'/home/killfm/Videos/jreAJ.mp4'
    outputmoviefile = f'/home/killfm/Videos/deleteme105.mp4'
    h, m, s = 1, 14, 36
    duration = 59
    timestart = h*60*60 + m*60 + s
    timeend = timestart + duration
    cutclip(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
        # force=True,
    )

# Convert wav to mp3
if __name__ == '__main__' and True:
    force_overwrite = True
    inputmoviefile = f'/home/killfm/Videos/alex jones x neon genesis evangelion.mp4'
    outputmoviefile = f'/home/killfm/Videos/aj/aj_evangelion.mp4'
    mp3outpath = outputmoviefile.replace('.mp4', '.mp3')
    wavoutpath = mp3outpath.replace('.mp3', '.wav')
    h, m, s = 0, 0, 0
    duration = 122
    timestart = h*60*60 + m*60 + s
    timeend = timestart + duration
    cutclip(
        inputmoviefile,
        outputmoviefile,
        timestart=timestart,
        timeend=timeend,
        force=force_overwrite,
    )
    write_clip_audio_to_file(
        outputmoviefile,
        mp3outpath,
        force=force_overwrite,
    )
    mp3towav(
        mp3outpath,
        wavoutpath,
        force=force_overwrite,
    )
    asr_whisper(
        mp3outpath,
    )
