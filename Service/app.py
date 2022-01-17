from fastapi import FastAPI
from fastapi.responses import JSONResponse
from celery.result import AsyncResult

from celery_task_app.tasks import predict_text_analytics_single
from models import Input_Data_For_Generation

app = FastAPI(debug=True)


@app.post('/text_analytics/predict')
async def generate(input_data_for_generation: Input_Data_For_Generation):
    """Create celery prediction task. Return task_id to client in order to retrieve result"""
    task_id = predict_text_analytics_single.delay(input_data_for_generation.text_body)
    return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})


@app.get('/text_analytics/result/{task_id}')
async def generate_result(task_id):
    """Fetch result for given task_id"""
    task = AsyncResult(task_id)
    if not task.ready():
        return JSONResponse(status_code=202, content={'task_id': str(task_id), 'status': 'Processing'})
    result = task.get()
    return JSONResponse(status_code=200, content={'task_id': task_id, 'status': 'Success', 'generated_text': str(result)})