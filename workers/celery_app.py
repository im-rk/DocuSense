from celery import Celery
from config.settings import settings

celery_app = Celery(
    "docusense",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.tasks"]
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    worker_prefetch_multiplier=1,
)
