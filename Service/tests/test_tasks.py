import json
from models import Input_Data_For_Generation
    
    
def test_task_process_and_return_success(test_app):
    response = test_app.post(
        "/text_analytics/predict",
        data=json.dumps({"text_body": "text for test"})
    )
    content = response.json()
    task_id = content["task_id"]
    assert task_id
    while content["status"] == "Processing":
        response = test_app.get(f"/text_analytics/result/{task_id}")
        content = response.json()
    assert content["status"] == "Success"
    
    
    