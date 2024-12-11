from flask import Flask, request, jsonify, send_file, render_template
from werkzeug.utils import secure_filename
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Seam carving functions (copied from the provided script)
def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))
    energy_map = convolved.sum(axis=2)
    return energy_map

def crop_c(img, scale_c):
    r, c, _ = img.shape
    new_c = int(scale_c * c)
    for i in range(c - new_c):
        img = carve_column(img)
    return img

def carve_column(img):
    r, c, _ = img.shape
    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)
    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]
    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)
    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=int)
    for i in range(1, r):
        for j in range(0, c):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i - 1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]
            M[i, j] += min_energy
    return M, backtrack

@app.route('/')
def index():
    return render_template('homepage.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    axis = request.form['axis']
    scale = float(request.form['scale'])

    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_path = os.path.join(app.config['PROCESSED_FOLDER'], f'processed_{filename}')

        file.save(input_path)
        img = imread(input_path)

        if axis == 'c':
            processed_img = crop_c(img, scale)
        elif axis == 'r':
            processed_img = crop_c(np.rot90(img, 1, (0, 1)), scale)
            processed_img = np.rot90(processed_img, 3, (0, 1))
        else:
            return jsonify({"error": "Invalid axis"}), 400

        imwrite(output_path, processed_img)
        return send_file(output_path, mimetype='image/png')

    return jsonify({"error": "No file uploaded"}), 400

if __name__ == '__main__':
    app.run()