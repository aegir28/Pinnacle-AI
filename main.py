from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
import pandas as pd
from report_generator import generate_pinnacle_ai_report

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return "No file part in request"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
            eda_path, img_path, target, task, model_path = generate_pinnacle_ai_report(df)

            return render_template(
                'result.html',
                target_column=target,
                task_type=task,
                img_path=os.path.basename(img_path),
                report_file=os.path.basename(eda_path),
                model_file=os.path.basename(model_path)
            )
        except Exception as e:
            return f"Error during analysis: {str(e)}"
    else:
        return "Invalid file type. Please upload a CSV or Excel file."

@app.route('/download_report/<filename>')
def download_report(filename):
    return send_from_directory('generated_reports', filename, as_attachment=True)

@app.route('/download_model/<filename>')
def download_model(filename):
    return send_from_directory('generated_reports', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
