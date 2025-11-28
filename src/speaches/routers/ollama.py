import json
from typing import AsyncGenerator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.concurrency import run_in_threadpool

from speaches.dependencies import ExecutorRegistryDependency
from speaches.model_aliases import ModelId

router = APIRouter(tags=["ollama"])


class PullRequest(BaseModel):
    name: ModelId


class RunRequest(BaseModel):
    name: ModelId


async def download_model_streaming(
    executor_registry: ExecutorRegistryDependency, model_id: ModelId
) -> AsyncGenerator[str, None]:
    found_executor = None
    for executor in executor_registry.all_executors():
        if model_id in [
            model.id for model in executor.model_registry.list_remote_models()
        ]:
            found_executor = executor
            break

    if not found_executor:
        yield json.dumps({"error": f"model '{model_id}' not found"}) + "\n"
        return

    yield json.dumps({"status": f"pulling model '{model_id}'..."}) + "\n"

    was_downloaded = await run_in_threadpool(
        found_executor.model_registry.download_model_files_if_not_exist, model_id
    )

    if was_downloaded:
        yield json.dumps({"status": "success"}) + "\n"
    else:
        yield json.dumps({"status": f"Model '{model_id}' already exists."}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"


async def run_stream_generator(
    executor_registry: ExecutorRegistryDependency, model_id: ModelId
) -> AsyncGenerator[str, None]:
    # 1. Check if model is local
    for executor in executor_registry.all_executors():
        if model_id in [
            model.id for model in executor.model_registry.list_local_models()
        ]:
            yield json.dumps(
                {"status": f"Model '{model_id}' is already available."}
            ) + "\n"
            yield json.dumps({"status": "success"}) + "\n"
            return

    # 2. If not local, pull it
    async for chunk in download_model_streaming(executor_registry, model_id):
        yield chunk


@router.post("/api/pull")
async def pull_model(
    pull_request: PullRequest,
    executor_registry: ExecutorRegistryDependency,
) -> StreamingResponse:
    model_id = pull_request.name
    return StreamingResponse(
        download_model_streaming(executor_registry, model_id),
        media_type="application/x-ndjson",
    )


@router.post("/api/run")
async def run_model(
    run_request: RunRequest,
    executor_registry: ExecutorRegistryDependency,
) -> StreamingResponse:
    """
    Ensures a model is downloaded and available.

    This endpoint does not run the model directly. Instead, it prepares the model
    for use with the other API endpoints (e.g., /v1/audio/speech).
    """
    model_id = run_request.name
    return StreamingResponse(
        run_stream_generator(executor_registry, model_id),
        media_type="application/x-ndjson",
    )
