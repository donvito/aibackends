from pathlib import Path

from aibackends.tasks import create_task

task = create_task(
    "analyse-video-ad",
    runtime="llamacpp",
    model="gemma4-e2b",
)

brief_path = Path(__file__).parent.parent / "data" / "video_ad_brief.txt"
report = task.run(brief_path)

print(report.model_dump_json(indent=2))
