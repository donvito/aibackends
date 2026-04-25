from aibackends.core.registry import WorkflowSpec
from aibackends.schemas.video_ad import VideoAdReport
from aibackends.steps.enrich import VisionExtractor
from aibackends.steps.ingest import VideoIngestor
from aibackends.steps.process import AudioStripper, FrameExtractor, WhisperTranscriber
from aibackends.steps.validate import PydanticValidator
from aibackends.workflows._base import Pipeline


class VideoAdIntelligence(Pipeline):
    steps = [
        VideoIngestor(),
        FrameExtractor(sample_every_seconds=5),
        AudioStripper(),
        WhisperTranscriber(),
        VisionExtractor(schema=VideoAdReport, prompt="Analyse the video ad creative."),
        PydanticValidator(schema=VideoAdReport),
    ]


WORKFLOW_SPEC = WorkflowSpec(
    name="video-ad",
    workflow_factory=VideoAdIntelligence,
    aliases=("video_ad", "video-ad-intelligence", "video_ad_intelligence"),
)
