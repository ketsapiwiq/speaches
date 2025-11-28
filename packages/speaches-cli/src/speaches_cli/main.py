import json
import os

import httpx
import typer

app = typer.Typer()
registry_app = typer.Typer()
model_app = typer.Typer()
audio_app = typer.Typer()
audio_speech_app = typer.Typer()

SPEACHES_BASE_URL = os.getenv("SPEACHES_BASE_URL", "http://localhost:9000")
SPEACHES_OPENAI_BASE_URL = SPEACHES_BASE_URL + "/v1"
client = httpx.Client(base_url=SPEACHES_BASE_URL, timeout=httpx.Timeout(None))

MODELS_URL = f"{SPEACHES_OPENAI_BASE_URL}/models"
REGISTRY_URL = f"{SPEACHES_OPENAI_BASE_URL}/registry"


def dump_response(response: httpx.Response) -> None:
    if response.headers.get("Content-Type") == "application/json":
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(response.text)


@registry_app.command("ls")
def registry_ls(task: str | None = None) -> None:
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    response = client.get(REGISTRY_URL, params=params)
    dump_response(response)


@model_app.command("ls")
def models_ls(task: str | None = None) -> None:
    params: dict[str, str] = {}
    if task is not None:
        params["task"] = task
    response = client.get(MODELS_URL, params=params)
    dump_response(response)


@model_app.command("rm")
def model_rm(model_id: str) -> None:
    response = client.delete(f"{MODELS_URL}/{model_id}")
    dump_response(response)


@model_app.command("download")
def model_download(model_id: str) -> None:
    response = client.post(f"{MODELS_URL}/{model_id}")
    dump_response(response)


@app.command("pull")
def pull(
    model: str = typer.Argument(
        "istupakov/parakeet-tdt-0.6b-v3-onnx", help="Model ID to pull (default: parakeet-tdt-0.6b-v3-onnx)"
    ),
    background: bool = typer.Option(False, "--detach", "-d", help="Run in background"),  # noqa: FBT003
) -> None:
    """Pull a model from the registry"""
    _ = background  # Currently unused, but kept for API compatibility
    response = client.post(f"{MODELS_URL}/{model}")
    dump_response(response)


@app.command("run")
def run(
    model: str = typer.Argument(
        "istupakov/parakeet-tdt-0.6b-v3-onnx", help="Model ID to run (default: parakeet-tdt-0.6b-v3-onnx)"
    ),
    background: bool = typer.Option(True, "--detach", "-d", help="Run in background (default: true)"),  # noqa: FBT003
) -> None:
    """Run a model (alias for pull with detach by default)"""
    _ = background  # Currently unused, but kept for API compatibility
    response = client.post(f"{MODELS_URL}/{model}")
    dump_response(response)


app.add_typer(registry_app, name="registry")
app.add_typer(model_app, name="model")

if __name__ == "__main__":
    app()
