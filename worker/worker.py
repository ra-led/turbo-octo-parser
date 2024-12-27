import pika
import shutil
import json
import logging
import os
import importlib

# Configure logging
logger = logging.getLogger("worker")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("/logs/worker.log")
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

UPLOAD_FOLDER = '/data/uploads'
RESULT_FOLDER = '/data/results'
STATUS_FOLDER = '/data/status'
MODLIST_FILE = '/data/models'

PIPELINES = {}

for file_name in os.listdir('pipelines'):
    if not file_name.endswith('.py'):
        continue
    module_name = os.path.splitext(file_name)[0]
    module = importlib.import_module('pipelines.'+module_name)
    PIPELINES[module_name] = module

with open(MODLIST_FILE, 'w') as f:
    models = {}
    for model in PIPELINES.keys():
        models[model] = PIPELINES[model].ALLOWED_FORMATS
    print("models:", models)
    json.dump(models, f)


def callback(ch, method, properties, body):
    message = json.loads(body)
    task_id = message['task_id']
    pdf_path = message['pdf_path']
    model_name = message.get('model', list(PIPELINES.keys())[0])
    print("Using model", model_name)
    pipeline = PIPELINES[model_name]
    logger.info(f"Worker received task {task_id}")

    status_path = os.path.join(STATUS_FOLDER, f"{task_id}.status")
    # Update status to 'processing'
    with open(status_path, 'w') as status_file:
        status_file.write("processing")

    try:
        # Run the processing pipeline
        markdown_content = pipeline.run_pipeline(pdf_path)

        # Save the result

        TASK_RESULT_FOLDER = RESULT_FOLDER + f'/{task_id}'
        os.makedirs(TASK_RESULT_FOLDER)
        result_path = os.path.join(TASK_RESULT_FOLDER, "text.md")
        with open(result_path, 'w') as result_file:
            result_file.write(markdown_content)
        if os.path.exists('pages'):
            shutil.move('pages', TASK_RESULT_FOLDER)
        if os.path.exists('images'):
            shutil.move('images', TASK_RESULT_FOLDER)

        # Compress result
        shutil.make_archive(TASK_RESULT_FOLDER, 'zip', TASK_RESULT_FOLDER)

        # Update status to 'completed'
        with open(status_path, 'w') as status_file:
            status_file.write("completed")
        logger.info(f"Task {task_id} completed successfully")
    except Exception as e:
        # Update status to 'failed'
        with open(status_path, 'w') as status_file:
            status_file.write("failed")
        logger.error(f"Task {task_id} failed with error: {e}")
    finally:
        ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    # Setup RabbitMQ connection
    connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='rabbitmq', heartbeat=360))
    channel = connection.channel()
    channel.queue_declare(queue='pdf_tasks')
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='pdf_tasks', on_message_callback=callback)
    logger.info("Worker started consuming tasks")
    channel.start_consuming()


if __name__ == '__main__':
    main()
