from flask import render_template, jsonify 
import subprocess 
from app import app
import pickle
import ast

process_id = -1

data_root = "./app/static/data/"
live_gesture_prediction = data_root + "gesture.txt"
live_predictions_path = data_root + "predictions.txt"

@app.route("/flask", methods=['GET'])
@app.route("/", methods=['GET'])
def serve():
    return render_template('index.html',   
                        message = "Click start recording to begin predictions",
                        status="start", 
    )

@app.route("/gesture")
def gesture(): 
    with open(live_gesture_prediction, "r") as f: 
        predicted_gesture = f.read() 
    gesture_obj = {
        "gesture": predicted_gesture
    }
    return gesture_obj

@app.route("/predictions")
def predictions(): 
    with open(live_predictions_path, "r") as f: 
        predictions = ast.literal_eval(f.read())
    predictions_obj = {
        "predictions" : predictions
    }
    return jsonify(predictions_obj)

@app.route('/start')
def start():
    global process_id 
    print ("Starting recording")
    
    process_id = subprocess.Popen(["python3", "app/predict_simple.py"])
    return render_template('index.html',
                    message = "Started recording. Initialization takes about ten seconds",   
                    status="stop", 
    )

@app.route('/stop')
def stop(): 
    print("Stopping")
    process_id.kill()
    
    return render_template('index.html',   
                        status="start", 
    )
    

