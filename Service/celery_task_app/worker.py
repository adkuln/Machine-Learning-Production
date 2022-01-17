import os
from celery import Celery

BROKER_URI = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379")
BACKEND_URI = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379")

app = Celery(
    'celery_app',
    broker=BROKER_URI,
    backend=BACKEND_URI,
    include=['celery_task_app.tasks']
)