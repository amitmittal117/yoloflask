import cv2
from ultralytics import YOLO
import os
import tempfile
import logging
import time
from flask import Flask, request, send_file, jsonify

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load the model
model = YOLO("yolo11n.pt")

@app.route("/detect", methods=["POST"])
def detect_objects():
    logging.info("Received request for object detection")
    if "image" not in request.files:
        logging.warning("No image provided in request")
        return {"error": "No image provided"}, 400
    
    image_file = request.files["image"]
    
    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, image_file.filename)
    
    image_file.save(image_path)
    logging.info(f"Image saved at {image_path}")
    
    # Perform object detection on the image
    results = model(image_path)
    
    # Save the detected image with an updated filename
    name, ext = os.path.splitext(image_file.filename)
    updated_filename = f"{name}_updated{ext}"
    save_path = os.path.join(temp_dir, updated_filename)
    results[0].save(filename=save_path)
    logging.info(f"Processed image saved at {save_path}")
    
    return send_file(save_path, mimetype="image/jpeg")

@app.route("/detect_video", methods=["POST"])
def detect_video():
    logging.info("Received request for video detection")
    if "video" not in request.files:
        logging.warning("No video provided in request")
        return {"error": "No video provided"}, 400
    
    video_file = request.files["video"]

    # Create a temporary directory
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, video_file.filename)

    video_file.save(video_path)
    logging.info(f"Video saved at {video_path}")

    # Perform object detection on the video
    results = model(video_path)

    # Save the detected video with an updated filename
    name, ext = os.path.splitext(video_file.filename)
    if ext.lower() not in [".mp4", ".avi", ".mov"]:  # Ensure a supported video format
        ext = ".mp4"  # Default to MP4
    updated_filename = f"{name}_updated{ext}"
    save_path = os.path.join(temp_dir, updated_filename)
    if not results[0].save(filename=save_path):  # Check if OpenCV can save
        logging.error(f"Failed to save video. Trying alternative method.")
        save_path = os.path.join(temp_dir, f"{name}_updated.mp4")  # Default MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MP4 codec
        frame_size = (640, 480)  # Change based on your video resolution
        fps = 30  # Adjust based on the input video

        out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)
        
        for frame in results[0].frames:
            out.write(frame)
        
        out.release()

    results[0].save(filename=save_path)
    logging.info(f"Processed video saved at {save_path}")
    
    return send_file(save_path, mimetype="video/mp4")

@app.route("/train", methods=["POST"])
def train_model():
    logging.info("Received request to start training")
    data = request.json or {}
    dataset = data.get("dataset", "coco8.yaml")
    epochs = int(data.get("epochs", 10))
    imgsz = int(data.get("imgsz", 640))
    device = data.get("device", "cpu")
    
    logging.info(f"Starting training with dataset: {dataset}, epochs: {epochs}, imgsz: {imgsz}, device: {device}")
    start_time = time.time()
    
    try:
        train_results = model.train(
            data=dataset,
            epochs=epochs,
            imgsz=imgsz,
            device=device
        )
        model_summary_raw = model.info()
        if len(model_summary_raw) > 3:
            model_summary = {
                "layers": model_summary_raw[0],
                "parameters": model_summary_raw[1],
                "gradients": model_summary_raw[2],
                "GFLOPs": model_summary_raw[3]
            }
        end_time = time.time()
        total_time = round(end_time - start_time, 2)
        logging.info(f"Training completed successfully in {total_time} seconds")
        result = {}
        if train_results:
            result = train_results.speed
            result['task'] = train_results.task
            result['total_time'] = total_time
            result['model_summary'] = model_summary
        return jsonify({"message": "Training started successfully", "results": result})
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logging.info("Starting Flask server on port 5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
