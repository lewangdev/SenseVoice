from typing import Annotated
from fastapi import FastAPI, Form, File
from funasr import AutoModel
import os
from io import BytesIO
import dotenv
import logging
from postprocess_utils import rich_transcription_postprocess

LOGGER = logging.getLogger(__name__)
dotenv.load_dotenv()

DEFAULTS = {
    "MODEL_DIR": "iic/SenseVoiceSmall",
    "DEVICE": "cuda:0",
    "COMPUTE_TYPE": "float16",
    "BEAM_SIZE": 5,
    "VAD_FILTER": "true",
    "MIN_SILENCE_DURATION_MS": 50,
}


def get_env(key):
    return os.environ.get(key, DEFAULTS.get(key))


def get_int_env(key):
    return int(get_env(key))


def get_float_env(key):
    return float(get_env(key))


def get_bool_env(key):
    return get_env(key).lower() == 'true'

asr_model = AutoModel(model=get_env("MODEL_DIR"),
                  trust_remote_code=True, device=get_env("DEVICE"))

app = FastAPI()


@app.post("/v1/audio/transcriptions")
def create_transcription(file: Annotated[bytes, File()],
                         model: Annotated[str, Form()] = 'whipser-1',
                         language: Annotated[str | None, Form()] = 'zh',
                         prompt: Annotated[str | None, Form()] = None):


    res = asr_model.generate(
      input=file,
      cache={},
      language=language, # "zn", "en", "yue", "ja", "ko", "nospeech"
      use_itn=False,
      batch_size=64,
      merge_vad=False
    )

    return {
        "text": rich_transcription_postprocess(res[0]['text'])
    }