import math
from base64 import urlsafe_b64encode
from hashlib import sha512
from pathlib import Path
from tempfile import NamedTemporaryFile

from google.cloud import storage, speech
from pydub import AudioSegment

one_second = 1000
one_minute = one_second * 60

config = speech.types.RecognitionConfig(
    encoding=speech.enums.RecognitionConfig.AudioEncoding.FLAC,
    language_code='en-US',
    audio_channel_count=2,
    sample_rate_hertz=44100)


def trigger_transcriptions(speech_client, speech_config, bucket_name, blob_keys):
    for blob_key in blob_keys:
        audio = speech.types.RecognitionAudio(uri=f'gs://{bucket_name}/{blob_key}')
        yield speech_client.long_running_recognize(config=speech_config, audio=audio)


def upload_blobs(storage_client, root, bucket_name, recordings):
    for idx, recording in enumerate(recordings):
        bucket = storage_client.bucket(bucket_name)
        key = f"{root}/minute-{idx}.flac"
        with NamedTemporaryFile(suffix=".flac") as tf:
            recording.export(tf.name, format='flac')
            bucket.blob(key).upload_from_filename(tf.name)
        yield key


def split_sounds(sound, window_length, overlap_length):
    recording_count = math.ceil(len(sound) / window_length)
    for idx in range(recording_count):
        start = max(0, idx - 1) * window_length
        stop = (idx + 1) * window_length + overlap_length
        yield sound[start:stop]


def new_words(transcriptions, previous):
    return " ".join(transcriptions)


def write_out(output_path, operations):
    previous = None
    with open(output_path, 'w') as f:
        for minute, operation in enumerate(operations):
            transcriptions = [_.alternatives[0].transcript for _ in operation.result().results]
            h, m = divmod(minute, 60)
            f.write(f"{h}:{m:02d}, {new_words(transcriptions, previous)}\n")
            previous = transcriptions


def derive_hashed_name(sound_path):
    return urlsafe_b64encode(sha512(str(sound_path).encode()).digest())[:10],


def transcribe(project_name, bucket_name, sound_path, transcript_file):
    storage_client = storage.Client(project=project_name)
    speech_client = speech.SpeechClient()
    recordings = list(split_sounds(AudioSegment.from_file(sound_path), one_minute, one_second * 10))
    uploaded_blobs = list(upload_blobs(storage_client, derive_hashed_name(sound_path), bucket_name, recordings))
    operations = list(trigger_transcriptions(speech_client, config, bucket_name, uploaded_blobs))
    write_out(transcript_file, operations)
