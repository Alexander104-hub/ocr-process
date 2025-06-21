import logging
from flask import Flask, render_template, request, jsonify, url_for
import os
import json
import redis
import uuid
from werkzeug.utils import secure_filename
import base64
from pathlib import Path

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["OUTPUT_FOLDER"] = "output"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}
app.logger.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)


REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        id = str(uuid.uuid4())
        filename_with_id = f"{id}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename_with_id)
        file.save(filepath)

        Path(app.config["OUTPUT_FOLDER"]).mkdir(exist_ok=True)

        task_data = {
            "id": id,
            "filename": filename_with_id,
            "filepath": filepath,
            "output_dir": os.path.abspath(app.config["OUTPUT_FOLDER"]),
        }
        redis_client.rpush("ocr_tasks", json.dumps(task_data))

        return jsonify(
            {
                "success": True,
                "filename": filename,
                "id": id,
                "redirect": url_for("view_results", id=id, filename=filename),
            }
        )

    return jsonify({"error": "File type not allowed"}), 400


@app.route("/results/<id>/<filename>")
def view_results(id, filename):
    image_path = None
    for file in os.listdir(app.config["UPLOAD_FOLDER"]):
        if file.startswith(id):
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file)
            break

    img_data = None

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode("utf-8")

    results_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{id}_results.json")

    task_status = redis_client.get(f"task_status:{id}")
    status = "processing"

    if task_status:
        status = task_status.decode("utf-8")

    ocr_results = []
    if status == "completed" and os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            ocr_results = json.load(f)

    return render_template(
        "results.html",
        filename=filename,
        image_data=img_data,
        ocr_results=ocr_results,
        ocr_results_serialized=json.dumps(ocr_results),
        status=status,
        id=id,
    )


@app.route("/api/task_status/<id>")
def get_task_status(id):
    status = redis_client.get(f"task_status:{id}")
    if status:
        return jsonify({"status": status.decode("utf-8")})
    return jsonify({"status": "unknown"})


@app.route("/api/ocr_results/<id>")
def get_ocr_results(id):
    results_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{id}_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify([])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
