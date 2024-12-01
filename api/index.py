import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model
model = load_model("waste_classifier.h5", compile=False)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file is uploaded
        if 'file' not in request.files:
            return render_template("index.html", result="No file uploaded.")
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Predict the image category
            category, confidence = predict_image(file_path)
            result = f"The image is classified as: {category} with confidence: {confidence:.2f}"
            return render_template("index.html", result=result, image_path=file_path)
        else:
            return render_template("index.html", result="Invalid file type.")
    
    return render_template("index.html")

def predict_image(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the category
    prediction = model.predict(img_array)[0][0]
    category = "Biodegradable" if prediction < 0.5 else "Non-Biodegradable"
    confidence = 1 - prediction if category == "Biodegradable" else prediction
    confidence *= 100
    return category, confidence

if __name__ == "__main__":
    app.run()
