import os
import json
import time
import luigi
import redis

from tasks.tasks import OCRPipeline
from luigi_logger import logger


REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

LUIGI_SCHEDULER_HOST = os.environ.get('LUIGI_SCHEDULER_HOST', 'localhost')
LUIGI_SCHEDULER_PORT = int(os.environ.get('LUIGI_SCHEDULER_PORT', 8082))

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

def process_task(task_data):
    try:
        id = task_data['id']
        filepath = task_data['filepath']
        output_dir = task_data['output_dir']
        
        logger.info(f"Processing task {id} for file {filepath}")

        redis_client.set(f"task_status:{id}", "processing")
        result = luigi.build(
            [OCRPipeline(
                image_path=filepath,
                output_dir=output_dir,
                id=id
            )],
            scheduler_host=LUIGI_SCHEDULER_HOST,
            scheduler_port=LUIGI_SCHEDULER_PORT
        )
        
        if result:
            logger.info(f"Task {id} completed successfully")
            redis_client.set(f"task_status:{id}", "completed")

            results_path = os.path.join(output_dir, f"{id}_results.json")
            if not os.path.exists(results_path):
                with open(results_path, 'w', encoding='utf-8') as f:
                    json.dump({"message": "No text detected or processing failed"}, f)
        else:
            logger.error(f"Task {id} failed")
            redis_client.set(f"task_status:{id}", "failed")
            
    except Exception as e:
        logger.exception(f"Error processing task: {e}")
        if 'id' in task_data:
            redis_client.set(f"task_status:{task_data['id']}", "failed")

def main():
    logger.info("Starting Luigi Worker")

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    while True:
        try:
            task = redis_client.blpop('ocr_tasks', timeout=1)
            
            if task:
                _, task_json = task
                task_data = json.loads(task_json)
                process_task(task_data)

            time.sleep(0.1)
            
        except Exception as e:
            logger.exception(f"Error in main loop: {e}")
            time.sleep(1)

if __name__ == "__main__":
    main()
