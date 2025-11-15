from collections.abc import Generator
import logging
from pathlib import Path
import time
from typing import Literal

import huggingface_hub
from pydantic import BaseModel, computed_field

from speaches.api_types import Model
from speaches.audio import Audio
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import SpeechRequest, SpeechResponse
from speaches.hf_utils import (
    HfModelFilter,
    extract_language_list,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.tracing import traced_generator
from speaches.utils import async_to_sync_generator

SAMPLE_RATE = 24000  # XTTS v2 default sample rate
LIBRARY_NAME = "coqui"
TASK_NAME_TAG = "text-to-speech"
TAGS = {"speaches", "xtts", "coqui"}


class XTTSv2ModelFiles(BaseModel):
    config: Path
    vocab: Path
    model_dir: Path


class XTTSv2ModelVoice(BaseModel):
    name: str
    language: str

    @computed_field
    @property
    def id(self) -> str:
        return self.name


# Built-in XTTS v2 speakers from Coqui
XTTS_V2_SPEAKERS = [
    # English speakers
    XTTSv2ModelVoice(name="Claribel Dervla", language="en"),
    XTTSv2ModelVoice(name="Daisy Studious", language="en"),
    XTTSv2ModelVoice(name="Gracie Wise", language="en"),
    XTTSv2ModelVoice(name="Tammie Ema", language="en"),
    XTTSv2ModelVoice(name="Alison Dietlinde", language="en"),
    XTTSv2ModelVoice(name="Ana Florence", language="en"),
    XTTSv2ModelVoice(name="Annmarie Nele", language="en"),
    XTTSv2ModelVoice(name="Asya Anara", language="en"),
    XTTSv2ModelVoice(name="Brenda Stern", language="en"),
    XTTSv2ModelVoice(name="Gitta Nikolina", language="en"),
    XTTSv2ModelVoice(name="Henriette Usha", language="en"),
    XTTSv2ModelVoice(name="Sofia Hellen", language="en"),
    XTTSv2ModelVoice(name="Tammy Grit", language="en"),
    XTTSv2ModelVoice(name="Tanja Adelina", language="en"),
    XTTSv2ModelVoice(name="Vjollca Johnnie", language="en"),
    XTTSv2ModelVoice(name="Andrew Chipper", language="en"),
    XTTSv2ModelVoice(name="Badr Odhiambo", language="en"),
    XTTSv2ModelVoice(name="Dionisio Schuyler", language="en"),
    XTTSv2ModelVoice(name="Royston Min", language="en"),
    XTTSv2ModelVoice(name="Viktor Eka", language="en"),
    XTTSv2ModelVoice(name="Abrahan Mack", language="en"),
    XTTSv2ModelVoice(name="Adde Michal", language="en"),
    XTTSv2ModelVoice(name="Baldur Sanjin", language="en"),
    XTTSv2ModelVoice(name="Craig Gutsy", language="en"),
    XTTSv2ModelVoice(name="Damien Black", language="en"),
    XTTSv2ModelVoice(name="Gilberto Mathias", language="en"),
    XTTSv2ModelVoice(name="Ilkin Urbano", language="en"),
    XTTSv2ModelVoice(name="Kazuhiko Atallah", language="en"),
    XTTSv2ModelVoice(name="Ludvig Milivoj", language="en"),
    XTTSv2ModelVoice(name="Suad Qasim", language="en"),
    XTTSv2ModelVoice(name="Torcull Diarmuid", language="en"),
    XTTSv2ModelVoice(name="Viktor Menelaos", language="en"),
    XTTSv2ModelVoice(name="Zacharie Aimilios", language="en"),
    # Spanish speakers
    XTTSv2ModelVoice(name="Esperanza Curran", language="es"),
    XTTSv2ModelVoice(name="Viti Lucas", language="es"),
    XTTSv2ModelVoice(name="Beto", language="es"),
    XTTSv2ModelVoice(name="Jordi", language="es"),
    # French speakers
    XTTSv2ModelVoice(name="Denise", language="fr"),
    XTTSv2ModelVoice(name="Henri", language="fr"),
    # German speakers
    XTTSv2ModelVoice(name="Friedrich", language="de"),
    XTTSv2ModelVoice(name="Hokuspokus", language="de"),
    # Italian speakers
    XTTSv2ModelVoice(name="Alessio", language="it"),
    XTTSv2ModelVoice(name="Rachele", language="it"),
    # Portuguese speakers
    XTTSv2ModelVoice(name="Edmundo", language="pt"),
    XTTSv2ModelVoice(name="Raquel", language="pt"),
    # Polish speakers
    XTTSv2ModelVoice(name="Maciej", language="pl"),
    XTTSv2ModelVoice(name="Grzegorz", language="pl"),
    # Other languages
    XTTSv2ModelVoice(name="Sofia", language="tr"),
    XTTSv2ModelVoice(name="Maja", language="tr"),
    XTTSv2ModelVoice(name="Arkadiusz", language="pl"),
    XTTSv2ModelVoice(name="Afonso", language="pt"),
    XTTSv2ModelVoice(name="Ana", language="pt"),
    XTTSv2ModelVoice(name="Bogdan", language="pl"),
    XTTSv2ModelVoice(name="Celina", language="pl"),
    XTTSv2ModelVoice(name="Dominika", language="pl"),
    XTTSv2ModelVoice(name="Elżbieta", language="pl"),
    XTTSv2ModelVoice(name="Emilia", language="pl"),
    XTTSv2ModelVoice(name="Franciszek", language="pl"),
    XTTSv2ModelVoice(name="Gabriela", language="pl"),
    XTTSv2ModelVoice(name="Grzegorz", language="pl"),
    XTTSv2ModelVoice(name="Hanna", language="pl"),
    XTTSv2ModelVoice(name="Iga", language="pl"),
    XTTSv2ModelVoice(name="Jakub", language="pl"),
    XTTSv2ModelVoice(name="Jan", language="pl"),
    XTTSv2ModelVoice(name="Joanna", language="pl"),
    XTTSv2ModelVoice(name="Kacper", language="pl"),
    XTTSv2ModelVoice(name="Kamil", language="pl"),
    XTTSv2ModelVoice(name="Karolina", language="pl"),
    XTTSv2ModelVoice(name="Katarzyna", language="pl"),
    XTTSv2ModelVoice(name="Kornelia", language="pl"),
    XTTSv2ModelVoice(name="Krzysztof", language="pl"),
    XTTSv2ModelVoice(name="Laura", language="pl"),
    XTTSv2ModelVoice(name="Lena", language="pl"),
    XTTSv2ModelVoice(name="Lukasz", language="pl"),
    XTTSv2ModelVoice(name="Magda", language="pl"),
    XTTSv2ModelVoice(name="Malgorzata", language="pl"),
    XTTSv2ModelVoice(name="Marcelina", language="pl"),
    XTTSv2ModelVoice(name="Maria", language="pl"),
    XTTSv2ModelVoice(name="Marta", language="pl"),
    XTTSv2ModelVoice(name="Mateusz", language="pl"),
    XTTSv2ModelVoice(name="Michal", language="pl"),
    XTTSv2ModelVoice(name="Natalia", language="pl"),
    XTTSv2ModelVoice(name="Nina", language="pl"),
    XTTSv2ModelVoice(name="Olga", language="pl"),
    XTTSv2ModelVoice(name="Oliwia", language="pl"),
    XTTSv2ModelVoice(name="Patrycja", language="pl"),
    XTTSv2ModelVoice(name="Paulina", language="pl"),
    XTTSv2ModelVoice(name="Pawel", language="pl"),
    XTTSv2ModelVoice(name="Piotr", language="pl"),
    XTTSv2ModelVoice(name="Rafal", language="pl"),
    XTTSv2ModelVoice(name="Roksana", language="pl"),
    XTTSv2ModelVoice(name="Sandra", language="pl"),
    XTTSv2ModelVoice(name="Sebastian", language="pl"),
    XTTSv2ModelVoice(name="Sylwia", language="pl"),
    XTTSv2ModelVoice(name="Tomasz", language="pl"),
    XTTSv2ModelVoice(name="Urszula", language="pl"),
    XTTSv2ModelVoice(name="Weronika", language="pl"),
    XTTSv2ModelVoice(name="Wiktoria", language="pl"),
    XTTSv2ModelVoice(name="Zofia", language="pl"),
    XTTSv2ModelVoice(name="Łukasz", language="pl"),
]


class XTTSv2Model(Model):
    sample_rate: int
    voices: list[XTTSv2ModelVoice]


hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
    tags=TAGS,
)


logger = logging.getLogger(__name__)


class XTTSv2ModelRegistry(ModelRegistry):
    def list_remote_models(self) -> Generator[XTTSv2Model]:
        # XTTS v2 is a specific model from Coqui
        yield XTTSv2Model(
            id="coqui/XTTS-v2",
            created=1700000000,  # Approximate creation time
            owned_by="coqui",
            language=[
                "ar", "zh-cn", "cs", "nl", "en", "fr", "de", "hi", "hu", "it", 
                "ja", "ko", "pl", "pt", "ru", "es", "tr"
            ],
            task=TASK_NAME_TAG,
            sample_rate=SAMPLE_RATE,
            voices=XTTS_V2_SPEAKERS,
        )

    def list_local_models(self) -> Generator[XTTSv2Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if cached_repo_info.repo_id == "coqui/XTTS-v2":
                yield XTTSv2Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by="coqui",
                    language=[
                        "ar", "zh-cn", "cs", "nl", "en", "fr", "de", "hi", "hu", "it", 
                        "ja", "ko", "pl", "pt", "ru", "es", "tr"
                    ],
                    task=TASK_NAME_TAG,
                    sample_rate=SAMPLE_RATE,
                    voices=XTTS_V2_SPEAKERS,
                )

    def get_model_files(self, model_id: str) -> XTTSv2ModelFiles:
        model_files = list(list_model_files(model_id))
        
        config_file_path = next(file_path for file_path in model_files if file_path.name == "config.json")
        vocab_file_path = next(file_path for file_path in model_files if file_path.name == "vocab.json")
        model_dir = config_file_path.parent

        return XTTSv2ModelFiles(
            config=config_file_path,
            vocab=vocab_file_path,
            model_dir=model_dir,
        )

    def download_model_files(self, model_id: str) -> None:
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, 
            repo_type="model", 
            allow_patterns=["config.json", "vocab.json", "*.pth", "README.md"]
        )


xtts_v2_model_registry = XTTSv2ModelRegistry(hf_model_filter=hf_model_filter)


class XTTSv2ModelManager(BaseModelManager):
    def __init__(self, ttl: int) -> None:
        super().__init__(ttl)

    def _load_fn(self, model_id: str):
        try:
            from TTS.api import TTS
        except ImportError as e:
            raise ImportError(
                "XTTS v2 requires the coqui-tts package. Install with: pip install coqui-tts"
            ) from e

        # Load XTTS v2 model
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        return tts

    @traced_generator()
    def handle_speech_request(
        self,
        request: SpeechRequest,
        **_kwargs,
    ) -> SpeechResponse:
        if request.speed < 0.5 or request.speed > 2.0:
            msg = f"Speed must be between 0.5 and 2.0, got {request.speed}"
            raise ValueError(msg)

        # Check if voice is supported
        supported_voices = [v.name for v in XTTS_V2_SPEAKERS]
        if request.voice not in supported_voices:
            msg = f"Voice '{request.voice}' is not supported. Supported voices: {supported_voices}"
            raise ValueError(msg)

        # Get language for the voice
        voice_language = next(v.language for v in XTTS_V2_SPEAKERS if v.name == request.voice)

        with self.load_model(request.model) as tts:
            start = time.perf_counter()
            
            # Generate audio using the built-in speaker
            audio_data = tts.tts(
                text=request.text,
                speaker=request.voice,
                language=voice_language,
                speed=request.speed
            )
            
            # Convert to numpy array and yield
            import numpy as np
            audio_array = np.array(audio_data, dtype=np.float32)
            
            yield Audio(audio_array, sample_rate=SAMPLE_RATE)

        logger.info(f"Generated audio for {len(request.text)} characters in {time.perf_counter() - start}s")