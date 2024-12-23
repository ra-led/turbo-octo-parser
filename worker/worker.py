import pika
import json
import logging
import os
from pipeline import run_pipeline

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

def callback(ch, method, properties, body):
    message = json.loads(body)
    task_id = message['task_id']
    pdf_path = message['pdf_path']
    logger.info(f"Worker received task {task_id}")

    status_path = os.path.join(STATUS_FOLDER, f"{task_id}.status")
    # Update status to 'processing'
    with open(status_path, 'w') as status_file:
        status_file.write("processing")

    try:
        # Run the processing pipeline
        markdown_content = run_pipeline(pdf_path)

        # Save the result

        result_path = os.path.join(RESULT_FOLDER, f"{task_id}.md")
        with open(result_path, 'w') as result_file:
            result_file.write(markdown_content)

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
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='rabbitmq'))
    channel = connection.channel()
    channel.queue_declare(queue='pdf_tasks')
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue='pdf_tasks', on_message_callback=callback)
    logger.info("Worker started consuming tasks")
    channel.start_consuming()

if __name__ == '__main__':
    main()

