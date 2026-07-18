# Docker (GPU)

The repository ships a `Dockerfile` that builds a CUDA-enabled image for GPU
clouds such as [RunPod](https://runpod.io) and [Modal](https://modal.com). It
compiles `llama-cpp-python` with CUDA kernels and installs the
`transformers`, `pdf`, `audio`, `video`, and `pii` extras on top of a
`nvidia/cuda` runtime base image.

## Build

```bash
docker build -t aibackends:cuda .
```

Build arguments:

| Arg | Default | Purpose |
| --- | --- | --- |
| `CUDA_ARCHITECTURES` | `75;80;86;89;90;100;120` | CUDA compute capabilities compiled into llama.cpp. Trim to just your GPU (e.g. `89` for L4/L40S/RTX 40xx) for a much faster build. |
| `AIBACKENDS_EXTRAS` | `llamacpp,transformers,pdf,audio,video,pii` | Package extras baked into the image. |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu128` | PyTorch wheel index. |
| `CUDA_VERSION` | `12.8.1` | Base image CUDA version. |

Example, a leaner image for an RTX 4090 with only the llama.cpp runtime:

```bash
docker build -t aibackends:4090 \
  --build-arg CUDA_ARCHITECTURES="89" \
  --build-arg AIBACKENDS_EXTRAS="llamacpp" .
```

## Run locally with a GPU

```bash
docker run --rm --gpus all \
  -v aibackends-models:/models \
  aibackends:cuda \
  aibackends task classify --input "invoice from ACME for $120" \
  --labels invoice,contract,receipt --runtime llamacpp --model gemma4-e2b
```

`HF_HOME` is set to `/models` inside the image, so mounting a volume there
persists downloaded models across runs. The llama.cpp runtime detects CUDA
and offloads all layers automatically; the transformers runtime uses
`accelerate` device mapping.

## RunPod

1. Build the image and push it to a registry RunPod can pull from
   (Docker Hub, GHCR, ...):

   ```bash
   docker build -t <user>/aibackends:cuda .
   docker push <user>/aibackends:cuda
   ```

2. Create a pod (or serverless endpoint) from that image. Attach a network
   volume and set its mount path to `/models` so model downloads survive pod
   restarts.
3. For an interactive pod, set the container start command to
   `sleep infinity`, then use the web terminal or SSH:

   ```bash
   aibackends pull gemma4-e2b --runtime llamacpp
   aibackends task summarize --input notes.txt --runtime llamacpp --model gemma4-e2b
   ```

For gated Hugging Face models, set an `HF_TOKEN` environment variable on the
pod.

## Modal

Modal can build the image directly from the Dockerfile:

```python
import modal

app = modal.App("aibackends")
image = modal.Image.from_dockerfile("Dockerfile")
volume = modal.Volume.from_name("aibackends-models", create_if_missing=True)


@app.function(image=image, gpu="L4", volumes={"/models": volume})
def summarize(text: str) -> str:
    from aibackends.models import GEMMA4_E2B
    from aibackends.runtimes import LLAMACPP
    from aibackends.tasks import SummarizeTask, create_task

    task = create_task(SummarizeTask, runtime=LLAMACPP, model=GEMMA4_E2B)
    return task.run(text)


@app.local_entrypoint()
def main() -> None:
    print(summarize.remote("Payments failed after the checkout deploy ..."))
```

Tip: pass `--build-arg CUDA_ARCHITECTURES=...` matching the `gpu=` type you
request (Modal supports build args via
`modal.Image.from_dockerfile(..., build_args={...})`) to cut build time.

## Image layout

- Python 3.12 virtualenv at `/opt/venv` (on `PATH`)
- `aibackends` CLI as the default command
- Runnable examples copied to `/app/examples`
- `ffmpeg` installed for the audio/video extras, cuDNN for `faster-whisper`
- Models cached under `/models` (`HF_HOME`)
