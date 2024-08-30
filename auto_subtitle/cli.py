import os
import ffmpeg
import whisper
import argparse
import warnings
import tempfile
from typing import Iterator, TextIO, List
from .utils import filename, str2bool, format_timestamp


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("video", nargs="+", type=str,
                        help="paths to video files to transcribe")
    parser.add_argument("--model", default="small",
                        choices=whisper.available_models(), help="name of the Whisper model to use")
    parser.add_argument("--output_dir", "-o", type=str,
                        default=".", help="directory to save the outputs")
    parser.add_argument("--output_srt", type=str2bool, default=False,
                        help="whether to output the .srt file along with the video files")
    parser.add_argument("--srt_only", type=str2bool, default=False,
                        help="only generate the .srt file and not create overlayed video")
    parser.add_argument("--verbose", type=str2bool, default=False,
                        help="whether to print out the progress and debug messages")

    parser.add_argument("--task", type=str, default="transcribe", choices=[
                        "transcribe", "translate"], help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--language", type=str, default="auto", choices=["auto","af","am","ar","as","az","ba","be","bg","bn","bo","br","bs","ca","cs","cy","da","de","el","en","es","et","eu","fa","fi","fo","fr","gl","gu","ha","haw","he","hi","hr","ht","hu","hy","id","is","it","ja","jw","ka","kk","km","kn","ko","la","lb","ln","lo","lt","lv","mg","mi","mk","ml","mn","mr","ms","mt","my","ne","nl","nn","no","oc","pa","pl","ps","pt","ro","ru","sa","sd","si","sk","sl","sn","so","sq","sr","su","sv","sw","ta","te","tg","th","tk","tl","tr","tt","uk","ur","uz","vi","yi","yo","zh"], 
    help="What is the origin language of the video? If unset, it is detected automatically.")

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)
    
    if model_name.endswith(".en"):
        warnings.warn(
            f"{model_name} is an English-only model, forcing English detection.")
        args["language"] = "en"
    # if translate task used and language argument is set, then use it
    elif language != "auto":
        args["language"] = language
        
        args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)

    # ... (rest of the model loading and language detection code remains the same)

    args = parser.parse_args().__dict__
    model_name: str = args.pop("model")
    output_dir: str = args.pop("output_dir")
    output_srt: bool = args.pop("output_srt")
    srt_only: bool = args.pop("srt_only")
    language: str = args.pop("language")
    
    os.makedirs(output_dir, exist_ok=True)


def get_subtitles(audio_paths: list, output_srt: bool, output_dir: str, transcribe: callable):
    subtitles_path = {}

    for path, audio_path in audio_paths.items():
        srt_path = output_dir if output_srt else tempfile.gettempdir()
        srt_path = os.path.join(srt_path, f"{filename(path)}.srt")
        
        print(f"Generating subtitles for {filename(path)}... This might take a while.")

        warnings.filterwarnings("ignore")
        result = transcribe(audio_path)
        warnings.filterwarnings("default")

        # Fallback to segment-level timing if word-level is not available
        subtitle_data = []
        for segment in result["segments"]:
            if "words" in segment:
                # Word-level timing available
                for word in segment["words"]:
                    subtitle_data.append({
                        "text": word["word"],
                        "start": word["start"],
                        "end": word["end"]
                    })
            else:
                # Fallback to segment-level timing
                subtitle_data.append({
                    "text": segment["text"],
                    "start": segment["start"],
                    "end": segment["end"]
                })

        with open(srt_path, "w", encoding="utf-8") as srt:
            write_styled_srt(subtitle_data, file=srt)

        subtitles_path[path] = srt_path

    return subtitles_path

def write_styled_srt(subtitle_data: List[dict], file: TextIO):
    for i, item in enumerate(subtitle_data, start=1):
        start_time = format_timestamp(item['start'], always_include_hours=True)
        end_time = format_timestamp(item['end'], always_include_hours=True)
        styled_text = f"{{\\fs40\\c&HFFFF00&}}{item['text']}{{\\fs24\\c&HFFFFFF&}}"
        
        print(
            f"{i}\n"
            f"{start_time} --> {end_time}\n"
            f"{styled_text}\n",
            file=file,
            flush=True,
        )
        
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

if __name__ == '__main__':
    main()
