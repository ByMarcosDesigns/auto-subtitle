import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import Iterator, TextIO, List
from .utils import filename, str2bool, format_timestamp
from subprocess import run, PIPE
import textwrap

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")
    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
                        help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    elif language != "auto":
        args["language"] = language

    model = whisper.load_model(model_name)
    audios = get_audio(args.pop("video"))
    subtitles = get_subtitles(
        audios, output_dir, lambda audio_path: model.transcribe(audio_path, **args)
    )

    for path, subtitle_data in subtitles.items():
        out_path = os.path.join(output_dir, f"{filename(path)}_subtitled.mp4")

        print(f"Adding subtitles to {filename(path)}...")

        add_subtitles_to_video(path, subtitle_data, out_path)

        print(f"Saved subtitled video to {os.path.abspath(out_path)}.")

def get_subtitles(audio_paths: dict, output_dir: str, transcribe: callable):
    subtitles = {}

    for path, audio_path in audio_paths.items():
        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        subtitle_data = []
        for segment in result["segments"]:
            subtitle_data.append({
                "text": segment["text"],
                "start": segment["start"],
                "end": segment["end"]
            })

        subtitles[path] = subtitle_data

    return subtitles

def get_audio(paths):
    temp_dir = tempfile.gettempdir()
    audio_paths = {}

    for path in paths:
        print(f"Extracting audio from {filename(path)}...")
        output_path = os.path.join(temp_dir, f"{filename(path)}.wav")

        ffmpeg.input(path).output(
            output_path,
            acodec="pcm_s16le", ac=1, ar="16k"
        ).run(quiet=True, overwrite_output=True)

        audio_paths[path] = output_path

    return audio_paths

def create_subtitle_image(text, width, height):
    wrapped_text = textwrap.fill(text, width=30)
    command = [
        "convert",
        "-size", f"{width}x{height}",
        "xc:none",
        "-gravity", "center",
        "-fill", "white",
        "-stroke", "black",
        "-strokewidth", "2",
        "-pointsize", "24",
        "-annotate", "0", wrapped_text,
        "-",
    ]
    result = run(command, stdout=PIPE, input=b"")
    return result.stdout

def add_subtitles_to_video(video_path, subtitle_data, output_path):
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    width = int(video_info['width'])
    height = int(video_info['height'])

    input_video = ffmpeg.input(video_path)
    
    subtitle_filter = ""
    for i, subtitle in enumerate(subtitle_data):
        start_time = subtitle['start']
        end_time = subtitle['end']
        duration = end_time - start_time
        
        subtitle_image = create_subtitle_image(subtitle['text'], width, height)
        temp_image_path = f"temp_subtitle_{i}.png"
        with open(temp_image_path, "wb") as f:
            f.write(subtitle_image)
        
        subtitle_filter += f"[0:v][s{i}]overlay=0:main_h-overlay_h:enable='between(t,{start_time},{end_time})'[v{i}];"
        input_video = ffmpeg.input(temp_image_path)
        subtitle_filter = f"[v{i}][{i+1}:v]" + subtitle_filter

    subtitle_filter = subtitle_filter.rstrip(";")
    output = ffmpeg.output(input_video, input_video.audio, output_path, vf=subtitle_filter)
    ffmpeg.run(output, overwrite_output=True)

    # Clean up temporary image files
    for i in range(len(subtitle_data)):
        os.remove(f"temp_subtitle_{i}.png")

if __name__ == '__main__':
    main()
