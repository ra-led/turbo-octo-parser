from typing import Annotated

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import uuid
import os
import aiofiles
import json
import logging
import pika

# Configure logging
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("/logs/api.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

UPLOAD_FOLDER = '/data/uploads'
RESULT_FOLDER = '/data/results'
STATUS_FOLDER = '/data/status'
MODLIST_FILE = '/data/models'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATUS_FOLDER, exist_ok=True)


def send_to_queue(task_id, pdf_path, model):
    connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='pdf_tasks')

    if model is None:
        message = json.dumps({'task_id': task_id, 'pdf_path': pdf_path})
    else:
        message = json.dumps({'task_id': task_id, 'pdf_path': pdf_path,
                              'model': model})

    channel.basic_publish(exchange='',
                          routing_key='pdf_tasks',
                          body=message)
    connection.close()
    logger.info(f"Task {task_id} sent to queue")


def check_file_extension_allowed(filename, model):
    with open(MODLIST_FILE) as f:
        models = json.load(f)

    filename = filename.lower()
    for ext in models[model]:
        if filename.endswith(ext.lower()):
            return True

    return False


@app.get("/models")
async def get_models():
    with open(MODLIST_FILE) as f:
        return json.load(f)


@app.post("/convert")
async def convert(file: UploadFile = File(...), model: Annotated[str, Form()] = 'turbo_octo_parser'):
    if not check_file_extension_allowed(file.filename, model):
        logger.error("Uploaded file extension is not allowed")
        raise HTTPException(status_code=400,
                            detail="Uploaded file extension is not allowed")

    task_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1]
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{task_id}.{ext}")

    # Save the uploaded PDF
    async with aiofiles.open(pdf_path, 'wb') as out_file:
        content = await file.read()
        await out_file.write(content)
    logger.info(f"Received and saved PDF for task {task_id}")

    # Initialize task status
    status_path = os.path.join(STATUS_FOLDER, f"{task_id}.status")
    with open(status_path, 'w') as status_file:
        status_file.write("pending")

    # Send task to queue
    send_to_queue(task_id, pdf_path, model)

    return {"task_id": task_id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    status_path = os.path.join(STATUS_FOLDER, f"{task_id}.status")
    if not os.path.exists(status_path):

        logger.error(f"Status requested for unknown task {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    with open(status_path, 'r') as status_file:
        status = status_file.read()

    return {"status": status}


@app.get("/result/{task_id}")
def get_result(task_id: str):
    result_path = os.path.join(RESULT_FOLDER, f"{task_id}.zip")
    status_path = os.path.join(STATUS_FOLDER, f"{task_id}.status")

    if not os.path.exists(status_path):
        logger.error(f"Result requested for unknown task {task_id}")
        raise HTTPException(status_code=404, detail="Task not found")

    with open(status_path, 'r') as status_file:
        status = status_file.read()

    if status != "completed":
        logger.warning(f"Result requested for task {task_id} with status {status}")
        raise HTTPException(status_code=400, detail="Result not available")

    if not os.path.exists(result_path):
        logger.error(f"Result file missing for task {task_id}")
        raise HTTPException(status_code=500, detail="Result file not found")

    return FileResponse(path=result_path, media_type='application/zip', filename=f"{task_id}.zip")


@app.get("/")
async def read_index():
    return FileResponse('ui/index.html')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
