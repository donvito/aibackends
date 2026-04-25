from pathlib import Path

from aibackends import configure
from aibackends.tasks import analyse_video_ad

configure(runtime="llamacpp", model="gemma4-e2b")

brief_path = Path(__file__).parent.parent / "data" / "video_ad_brief.txt"
report = analyse_video_ad(brief_path)

print(report.model_dump_json(indent=2))
