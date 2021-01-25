from flask import Flask, render_template, url_for, request, redirect
from flask_cors import CORS, cross_origin
import os
import model
from werkzeug.utils import secure_filename

app = Flask(__name__)
cors = CORS(app)
UPLOAD_FOLDER = 'landmark-app\static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def main_page():
    if request.method == 'POST':
        global file
        file = request.files['file']
        global image_path
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
        return redirect(url_for('prediction', filename = file.filename))
    return render_template('index.html')

@app.route('/prediction/<filename>')
def prediction(filename):
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    print(image_path)
    print(file.filename)
    classes, probs = model.pred_image(image_path)
    
    result = {
        'class_name': classes,
        'probs': probs,
        'image_path': file.filename
    }
    return render_template('predict.html', predictions=result)

app.run(host='0.0.0.0')
