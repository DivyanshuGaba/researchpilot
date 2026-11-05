from celery import Celery

from app.config import get_settings

celery_app = Celery("researchpilot", broker=get_settings().redis_url, include=["app.workers.tasks"])
celery_app.conf.update(
    task_default_queue="ingest",
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)
