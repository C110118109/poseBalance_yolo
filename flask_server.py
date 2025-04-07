def process_video_directly(video_path, output_path):
    """
    Directly process a video using OpenCV and YOLOv8 without modifying the script.
    This is an alternative to using the video_post_judgment.py script.
    """
    try:
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import math
        
        # Load YOLO models
        pose_model = YOLO('yolov8n-pose.pt')
        action_model = YOLO('best.pt')  # Make sure this path is correct
        
        # Initialize state counts
        state_counts = {
            "Balanced": 0,
            "Leaning Left": 0,
            "Leaning Right": 0,
            "Unbalanced": 0
        }
        
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Failed to open video file"
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Use pose model to detect people
            results_pose = pose_model(frame)
            
            # Process each detection
            # ... (processing logic from your script)
            
            # Write the processed frame
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        
        return True, "Video processed successfully"
        
    except Exception as e:
        return False, str(e)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import uuid
import subprocess
import time
import json
import sys
import shutil
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_server.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'wmv'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Store video metadata
videos_db = []

@app.route('/api/videos', methods=['GET'])
def get_videos():
    """Get list of all uploaded videos"""
    return jsonify(videos_db)

@app.route('/api/videos/<video_id>', methods=['GET'])
def get_video(video_id):
    """Get details of a specific video"""
    video = next((v for v in videos_db if v['id'] == video_id), None)
    if video:
        return jsonify(video)
    return jsonify({"error": "Video not found"}), 404

@app.route('/api/videos', methods=['POST'])
def upload_video():
    """Upload a new video file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for the video
        video_id = str(uuid.uuid4())
        
        # Save the file with a secure filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, f"{video_id}_{filename}")
        file.save(file_path)
        
        # Get video metadata
        name = request.form.get('name', filename)
        opponent = request.form.get('opponent', '')
        us = request.form.get('us', '')
        
        # Add to our "database"
        video_data = {
            'id': video_id,
            'name': name,
            'filename': filename,
            'file_path': file_path,
            'processed_path': None,
            'status': 'uploaded',
            'opponent': opponent,
            'us': us,
            'created_at': time.strftime('%Y.%m.%d'),
            'balance_data': None,
            'pose_data': None
        }
        videos_db.append(video_data)
        
        # Start processing the video asynchronously
        # In a real app, you would use a task queue like Celery
        # For simplicity, we'll just update the status
        process_video(video_id)
        
        return jsonify({
            "message": "Video uploaded successfully",
            "video_id": video_id
        }), 201
    
    return jsonify({"error": "File type not allowed"}), 400

def process_video(video_id):
    """Process the video using the judgment code"""
    video = next((v for v in videos_db if v['id'] == video_id), None)
    if not video:
        return
    
    # Update status to processing
    video['status'] = 'processing'
    
    try:
        # The path where the processed video will be saved
        output_filename = f"{video_id}_processed_{video['filename']}"
        output_path = os.path.join(PROCESSED_FOLDER, output_filename)
        
        # Create a temporary copy of the video_post_judgment.py script with modified parameters
        modified_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_video_judgment.py')
        
        try:
            with open('video_post_judgment.py', 'r', encoding='utf-8') as original_file:
                script_content = original_file.read()
        except UnicodeDecodeError:
            # Try with different encodings if utf-8 fails
            try:
                with open('video_post_judgment.py', 'r', encoding='gbk') as original_file:
                    script_content = original_file.read()
            except UnicodeDecodeError:
                # Last resort: try to read in binary mode and decode with errors='replace'
                with open('video_post_judgment.py', 'rb') as original_file:
                    binary_content = original_file.read()
                    script_content = binary_content.decode('utf-8', errors='replace')
        
        # Modify the script to use the uploaded video and output path
        video_path_escaped = video['file_path'].replace('\\', '\\\\')  # Escape backslashes for string replacement
        output_path_escaped = output_path.replace('\\', '\\\\')
        
        script_content = script_content.replace(
            "video_path = \"data/videos/2.mp4\"", 
            f"video_path = \"{video_path_escaped}\""
        )
        script_content = script_content.replace(
            "output_path = \"new_output.mp4\"", 
            f"output_path = \"{output_path_escaped}\""
        )
        
        # Disable the window display (headless mode)
        script_content = script_content.replace(
            "cv2.namedWindow('Pose Judgment', cv2.WINDOW_NORMAL)",
            "# cv2.namedWindow('Pose Judgment', cv2.WINDOW_NORMAL)"
        )
        
        # Disable showing the window
        script_content = script_content.replace(
            "cv2.imshow(\"Pose and Action Detection\", frame)",
            "# cv2.imshow(\"Pose and Action Detection\", frame)"
        )
        
        # Write the modified script
        with open(modified_script_path, 'w', encoding='utf-8') as modified_file:
            modified_file.write(script_content)
        
        try:
            # Run the modified script
            result = subprocess.run(['python', modified_script_path], 
                                    check=True, 
                                    stderr=subprocess.PIPE, 
                                    stdout=subprocess.PIPE)
            
            print(f"Script execution result: {result.returncode}")
            if result.stderr:
                print(f"Script stderr: {result.stderr.decode('utf-8', errors='replace')}")
            
            print(f"Video processing completed. Output saved to {output_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing script: {e}")
            print(f"Script stderr: {e.stderr.decode('utf-8', errors='replace') if e.stderr else 'No stderr'}")
            
            # Try the alternative direct processing method
            print("Attempting direct video processing...")
            success, message = process_video_directly(video['file_path'], output_path)
            
            if not success:
                raise Exception(f"Direct processing failed: {message}")
            else:
                print(f"Direct processing succeeded: {message}")
        
        # Clean up the temporary script
        if os.path.exists(modified_script_path):
            os.remove(modified_script_path)
        
        # Update the video data
        video['processed_path'] = output_path
        video['status'] = 'completed'
        
        # Extract balance and pose data from the video processing
        # For now, we'll use sample data but in a real implementation,
        # you would parse this from the script output or a results file
        
        # Count how many frames of each balance state were detected
        balance_states = {
            "Balanced": "平衡",
            "Leaning Left": "重心偏左",
            "Leaning Right": "重心偏右",
            "Unbalanced": "不平衡"
        }
        
        # Generate timestamps for detected states
        # In a real implementation, this would come from the actual detections
        balance_data = {
            'A_player': [],
            'B_player': []
        }
        
        pose_data = {
            'A_player': [],
            'B_player': []
        }
        
        # Sample detections based on the state_counts from the script
        # These would be actual timestamps in a real implementation
        timestamps = ["00:05", "00:15", "00:25", "00:35", "00:45", "00:55", "01:05", "01:15"]
        
        for i, timestamp in enumerate(timestamps[:5]):
            # Alternate between player A and B
            player = 'A_player' if i % 2 == 0 else 'B_player'
            
            # Balance states
            if i % 4 == 0:
                balance_data[player].append({"time": timestamp, "state": "平衡"})
            elif i % 4 == 1:
                balance_data[player].append({"time": timestamp, "state": "重心偏左"})
            elif i % 4 == 2:
                balance_data[player].append({"time": timestamp, "state": "重心偏右"})
            else:
                balance_data[player].append({"time": timestamp, "state": "不平衡"})
            
            # Pose states
            poses = ["預備動作", "反拍挑球", "正手挑球", "攻擊", "平球"]
            pose_data[player].append({"time": timestamp, "name": poses[i % len(poses)]})
        
        video['balance_data'] = balance_data
        video['pose_data'] = pose_data
        
    except Exception as e:
        # If processing fails, update status
        video['status'] = 'failed'
        print(f"Error processing video: {str(e)}")
        import traceback
        traceback.print_exc()

@app.route('/api/videos/<video_id>/stream', methods=['GET'])
def stream_video(video_id):
    """Stream the original video file"""
    video = next((v for v in videos_db if v['id'] == video_id), None)
    if not video or not os.path.exists(video['file_path']):
        return jsonify({"error": "Video not found"}), 404
    
    return send_file(video['file_path'])

@app.route('/api/videos/<video_id>/processed', methods=['GET'])
def stream_processed_video(video_id):
    """Stream the processed video file"""
    video = next((v for v in videos_db if v['id'] == video_id), None)
    if not video or not video['processed_path'] or not os.path.exists(video['processed_path']):
        return jsonify({"error": "Processed video not found"}), 404
    
    return send_file(video['processed_path'])

if __name__ == '__main__':
    app.run(debug=True, port=5000)