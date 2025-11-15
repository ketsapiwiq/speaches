from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import huggingface_hub
import numpy as np
import openai.types.audio
from opentelemetry import trace
from pydantic import BaseModel
import whisper
from whisper.tokenizer import LANGUAGES

from speaches.api_types import Model
from speaches.executors.shared.base_model_manager import BaseModelManager
from speaches.executors.shared.handler_protocol import (
    NonStreamingTranscriptionResponse,
    StreamingTranscriptionEvent,
    TranscriptionRequest,
    TranslationRequest,
    TranslationResponse,
)
from speaches.executors.silero_vad_v5 import merge_segments
from speaches.hf_utils import (
    HfModelFilter,
    get_cached_model_repos_info,
    get_model_card_data_from_cached_repo_info,
    list_model_files,
)
from speaches.model_registry import ModelRegistry
from speaches.text_utils import format_as_srt, format_as_vtt
from speaches.tracing import traced, traced_generator

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from speaches.config import WhisperConfig
    from speaches.routers.stt import ResponseFormat


LIBRARY_NAME = "pytorch"
TASK_NAME_TAG = "automatic-speech-recognition"

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

hf_model_filter = HfModelFilter(
    library_name=LIBRARY_NAME,
    task=TASK_NAME_TAG,
)


class WhisperModelFiles(BaseModel):
    model: Path


class WhisperModelRegistry(ModelRegistry[Model, WhisperModelFiles]):
    def list_remote_models(self) -> Generator[Model]:
        models = huggingface_hub.list_models(**self.hf_model_filter.list_model_kwargs(), cardData=True)
        for model in models:
            assert model.created_at is not None and model.card_data is not None, model
            yield Model(
                id=model.id,
                created=int(model.created_at.timestamp()),
                owned_by=model.id.split("/")[0],
                language=list(LANGUAGES.keys()),
                task=TASK_NAME_TAG,
            )

    def list_local_models(self) -> Generator[Model]:
        cached_model_repos_info = get_cached_model_repos_info()
        for cached_repo_info in cached_model_repos_info:
            model_card_data = get_model_card_data_from_cached_repo_info(cached_repo_info)
            if model_card_data is None:
                continue
            if self.hf_model_filter.passes_filter(cached_repo_info.repo_id, model_card_data):
                yield Model(
                    id=cached_repo_info.repo_id,
                    created=int(cached_repo_info.last_modified),
                    owned_by=cached_repo_info.repo_id.split("/")[0],
                    language=list(LANGUAGES.keys()),
                    task=TASK_NAME_TAG,
                )

    def get_model_files(self, model_id: str) -> WhisperModelFiles:
        model_files = list(list_model_files(model_id))
        model_file_path = next(file_path for file_path in model_files if file_path.name.endswith(".pt"))
        return WhisperModelFiles(model=model_file_path)

    def download_model_files(self, model_id: str) -> None:
        allow_patterns = ["*.pt", "*.json"]
        _model_repo_path_str = huggingface_hub.snapshot_download(
            repo_id=model_id, repo_type="model", allow_patterns=[*allow_patterns, "README.md"]
        )


whisper_model_registry = WhisperModelRegistry(hf_model_filter=hf_model_filter)


class WhisperModelManager(BaseModelManager[whisper.Whisper]):
    def __init__(self, ttl: int, whisper_config: WhisperConfig) -> None:
        super().__init__(ttl)
        self.whisper_config = whisper_config

    def _load_fn(self, model_id: str) -> whisper.Whisper:
        model_files = whisper_model_registry.get_model_files(model_id)
        model = whisper.load_model(
            str(model_files.model),
            device=self.whisper_config.inference_device,
        )
        return model

    @traced()
    def handle_non_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> NonStreamingTranscriptionResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        timelog_start = time.perf_counter()
        with self.load_model(request.model) as model:
            clip_timestamps = merge_segments(
                request.speech_segments,
                request.vad_options,
            )

            decode_options = {
                "task": "transcribe",
                "language": request.language,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "word_timestamps": "word" in request.timestamp_granularities,
            }

            result = model.transcribe(
                request.audio.data,
                **decode_options,
            )

            res = result_to_transcription_response(
                result,
                response_format=request.response_format,
            )
            logger.info(
                f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - timelog_start} seconds"
            )
            return res

    @traced_generator()
    def handle_streaming_transcription_request(
        self,
        request: TranscriptionRequest,
        **_kwargs,
    ) -> Generator[StreamingTranscriptionEvent]:
        timelog_start = time.perf_counter()
        with self.load_model(request.model) as model:
            clip_timestamps = merge_segments(
                request.speech_segments,
                request.vad_options,
            )

            decode_options = {
                "task": "transcribe",
                "language": request.language,
                "prompt": request.prompt,
                "temperature": request.temperature,
                "word_timestamps": "word" in request.timestamp_granularities,
            }

            result = model.transcribe(
                request.audio.data,
                **decode_options,
            )

            for segment in result["segments"]:
                yield openai.types.audio.TranscriptionTextDeltaEvent(
                    type="transcript.text.delta", delta=segment["text"], logprobs=None
                )

            yield openai.types.audio.TranscriptionTextDoneEvent(
                type="transcript.text.done", text=result["text"], logprobs=None
            )
        logger.info(
            f"Transcribed {request.audio.duration} seconds of audio in {time.perf_counter() - timelog_start} seconds"
        )

    def handle_transcription_request(
        self, request: TranscriptionRequest, **kwargs
    ) -> NonStreamingTranscriptionResponse | Generator[StreamingTranscriptionEvent]:
        if request.stream:
            return self.handle_streaming_transcription_request(request, **kwargs)
        else:
            return self.handle_non_streaming_transcription_request(request, **kwargs)

    @traced()
    def handle_translation_request(
        self,
        request: TranslationRequest,
        **_kwargs,
    ) -> TranslationResponse:
        if request.response_format == "diarized_json":
            raise NotImplementedError(
                f"'{request.response_format}' response format is not supported for '{request.model}' model."
            )
        with self.load_model(request.model) as model:
            decode_options = {
                "task": "translate",
                "prompt": request.prompt,
                "temperature": request.temperature,
            }

            result = model.transcribe(
                request.audio.data,
                **decode_options,
            )

            return result_to_translation_response(
                result,
                response_format=request.response_format,
            )


def result_to_transcription_response(
    result: dict,
    response_format: ResponseFormat,
) -> NonStreamingTranscriptionResponse:
    match response_format:
        case "text":
            return result["text"], "text/plain"
        case "json":
            return openai.types.audio.Transcription(
                text=result["text"],
            )

        case "verbose_json":
            return openai.types.audio.TranscriptionVerbose(
                language=result.get("language", ""),
                duration=result.get("duration", 0.0),
                text=result["text"],
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=segment["id"],
                        seek=segment.get("seek", 0),
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                        tokens=segment.get("tokens", []),
                        temperature=segment.get("temperature", 0.0),
                        avg_logprob=segment.get("avg_logprob", 0.0),
                        compression_ratio=segment.get("compression_ratio", 0.0),
                        no_speech_prob=segment.get("no_speech_prob", 0.0),
                    )
                    for segment in result.get("segments", [])
                ],
                words=[
                    openai.types.audio.TranscriptionWord(
                        start=word["start"],
                        end=word["end"],
                        word=word["word"],
                    )
                    for segment in result.get("segments", [])
                    for word in segment.get("words", [])
                ]
                if result.get("segments", [{}])[0].get("words")
                else None,
            )

        case "vtt":
            return "".join(
                format_as_vtt(segment["text"], segment["start"], segment["end"], i)
                for i, segment in enumerate(result.get("segments", []))
            ), "text/vtt"

        case "srt":
            return "".join(
                format_as_srt(segment["text"], segment["start"], segment["end"], i)
                for i, segment in enumerate(result.get("segments", []))
            ), "text/plain"


def result_to_translation_response(
    result: dict,
    response_format: ResponseFormat,
) -> TranslationResponse:
    match response_format:
        case "text":
            return result["text"], "text/plain"
        case "json":
            return openai.types.audio.Translation(
                text=result["text"],
            )

        case "verbose_json":
            return openai.types.audio.TranslationVerbose(
                language=result.get("language", ""),
                duration=result.get("duration", 0.0),
                text=result["text"],
                segments=[
                    openai.types.audio.TranscriptionSegment(
                        id=segment["id"],
                        seek=segment.get("seek", 0),
                        start=segment["start"],
                        end=segment["end"],
                        text=segment["text"],
                        tokens=segment.get("tokens", []),
                        temperature=segment.get("temperature", 0.0),
                        avg_logprob=segment.get("avg_logprob", 0.0),
                        compression_ratio=segment.get("compression_ratio", 0.0),
                        no_speech_prob=segment.get("no_speech_prob", 0.0),
                    )
                    for segment in result.get("segments", [])
                ],
            )

        case "vtt":
            return "".join(
                format_as_vtt(segment["text"], segment["start"], segment["end"], i)
                for i, segment in enumerate(result.get("segments", []))
            ), "text/vtt"

        case "srt":
            return "".join(
                format_as_srt(segment["text"], segment["start"], segment["end"], i)
                for i, segment in enumerate(result.get("segments", []))
            ), "text/plain"
