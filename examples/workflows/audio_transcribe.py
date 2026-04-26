"""Audio transcription workflow.

Reads ``examples/data/audio/audio1.mp3`` and transcribes it locally with
Whisper via the built-in ``WhisperTranscriber`` step.

Requires:
    pip install 'aibackends[audio]'
"""

from __future__ import annotations

import sys
from pathlib import Path

from aibackends.core.exceptions import AIBackendsError
from aibackends.steps.ingest import AudioIngestor
from aibackends.steps.process import WhisperTranscriber
from aibackends.workflows import Pipeline


class AudioTranscriptionWorkflow(Pipeline):
    steps = [
        AudioIngestor(),
        WhisperTranscriber(model_name="base"),
    ]


def main() -> None:
    try:
        audio_path = Path(__file__).parent.parent / "data" / "audio" / "audio1.mp3"
        workflow = AudioTranscriptionWorkflow()
        result = workflow.run(audio_path)

        transcript = result.get("transcript", "")
        print(f"Source: {result.get('path', '')}", flush=True)
        print("Whisper model: base\n", flush=True)
        print(transcript, flush=True)
    except KeyboardInterrupt:
        print("Example cancelled by user.", file=sys.stderr)
        raise SystemExit(130) from None
    except AIBackendsError as exc:
        print(f"Example failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
