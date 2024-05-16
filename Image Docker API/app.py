import os
import cv2
import pickle
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from numpy import asarray
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify, render_template
import tensorflow as tf

gpu_devices = tf.config.experimental.list_physical_devices('GPU')

if gpu_devices:
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)

    # Set per-process GPU memory fraction
    tf.config.experimental.set_virtual_device_configuration(
        gpu_devices[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=int(0.5 * 1024))]
    )
else:
    print("No GPU devices found.")
app = Flask(__name__)

# Load pre-trained VGGFace model
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')

# Load precomputed feature embeddings and filenames
feature_list = np.array(pickle.load(open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join("pklfile", 'FinalFilenames.pkl'), 'rb'))


def feature_extractor(img_data, model):
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def extract_faces_mtcnn(img_data, required_size=(224, 224)):
    detector = MTCNN()
    img = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
    converted_image = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    faces = detector.detect_faces(converted_image)
    if not faces:
        return {'status': 'Empty', 'Tracking Number': "Face not Found"}

    face_images = []
    for face_info in faces:
        x, y, w, h = face_info['box']
        x, y = max(x, 0), max(y, 0)
        face = img_data[y:y + h, x:x + w]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)
        face_images.append(face_array)

    return {'status': 'Success', 'faces': face_images}


def process_image(img_data):
    result = feature_extractor(img_data, model)
    similarity = cosine_similarity(result.reshape(1, -1), feature_list)
    max_similarity = np.max(similarity)

    if max_similarity >= 0.60:
        index = np.argmax(similarity)
        # print(filenames[index].split(".")[0])
        return {'status': 'Success', 'Tracking Number': filenames[index].split(".")[0]}
    else:
        return {'status': 'Success', 'Tracking Number': "Unknown"}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    img_stream = file.read()
    nparr = np.frombuffer(img_stream, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result_faces = extract_faces_mtcnn(img_np)

    if result_faces['status'] == 'Empty':
        result_faces = [result_faces]
        return jsonify(result_faces), 200
    else:
        # Process each detected face
        results = []
        for face_image in result_faces['faces']:
            result = process_image(face_image)
            results.append(result)
        print(results)
        return jsonify(results), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
