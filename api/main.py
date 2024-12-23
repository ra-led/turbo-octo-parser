from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import uuid
import os
import aiofiles
import asyncio
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

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(STATUS_FOLDER, exist_ok=True)

def send_to_queue(task_id, pdf_path):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='pdf_tasks')

    message = json.dumps({'task_id': task_id, 'pdf_path': pdf_path})
    channel.basic_publish(exchange='',
                          routing_key='pdf_tasks',
                          body=message)
    connection.close()
    logger.info(f"Task {task_id} sent to queue")

@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        logger.error("Uploaded file is not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    task_id = str(uuid.uuid4())
    pdf_path = os.path.join(UPLOAD_FOLDER, f"{task_id}.pdf")

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
    send_to_queue(task_id, pdf_path)

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
    result_path = os.path.join(RESULT_FOLDER, f"{task_id}/result.zip")
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

