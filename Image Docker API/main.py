import os
import cv2
import pickle
import shutil
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from numpy import asarray
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
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


def feature_extractor(img_path):
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


def extract_faces_mtcnn(input_folder, output_folder, required_size=(224, 224)):
    detector = MTCNN()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Error reading image: {img_path}")
            continue

        print(img.shape)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(img)
        ConvertedImage = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)

        faces = detector.detect_faces(ConvertedImage)
        if faces is None or not faces:
            continue
        else:
            for i, face_info in enumerate(faces):
                x, y, w, h = face_info['box']
                x, y = max(x, 0), max(y, 0)
                face = img[y:y + h, x:x + w]
                image_obj = Image.fromarray(face)
                image_obj = image_obj.resize(required_size)
                face_array = asarray(image_obj)
                face_filename = f"{filename}"
                output_path = os.path.join(output_folder, face_filename)
                cv2.imwrite(output_path, face_array)


def delete_files_in_folder(folder_path):
    print(folder_path)
    try:
        files = os.listdir(folder_path)
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")


def new_pickle_file(new_image_dir, update_face_dir):

    os.makedirs("pklfile", exist_ok=True)

    # Load existing pickle files if they exist, otherwise initialize empty lists
    try:
        existing_filenames = pickle.load(open(os.path.join("pklfile", 'FinalFilenames.pkl'), 'rb'))
    except (FileNotFoundError, EOFError):
        existing_filenames = []

    try:
        existing_features = pickle.load(open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'rb'))
    except (FileNotFoundError, EOFError):
        existing_features = []

    image_files = os.listdir(new_image_dir)

    if len(image_files) == 0:
        # No valid image files found in the directory.
        return

    # Filter out already processed files
    new_files = [file for file in image_files if file not in existing_filenames]

    if not new_files:
        # All files have already been processed
        return

    extract_faces_mtcnn(new_image_dir, update_face_dir)
    filenames = []
    features = []

    image_files = os.listdir(update_face_dir)
    if len(image_files) == 0:
        # No valid image files found in the update face directory.
        return

    # Filter out already processed files
    new_files = [file for file in image_files if file not in existing_filenames]

    if not new_files:
        # All files have already been processed
        return

    for person_file_name in new_files:
        filenames.append(person_file_name)

    # Update the existing pickle files with new data
    existing_filenames.extend(filenames)

    for file in filenames:
        features.append(feature_extractor(os.path.join(update_face_dir, file)))

    # Convert features to numpy array
    features = np.array(features)

    # Combine new data with existing data
    if len(existing_features) == 0:
        updated_features = features
    else:
        updated_features = np.concatenate((existing_features, features), axis=0)

    pickle.dump(existing_filenames, open(os.path.join("pklfile", 'FinalFilenames.pkl'), 'wb'))
    pickle.dump(updated_features, open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'wb'))

    # Clean up the new_pickle and new_extract_face directories
    # for folder in [update_face_dir]:
    #     delete_files_in_folder(folder)
    shutil.rmtree(update_face_dir)


# Example usage
new_pickle_file("C:/Users/HP/Desktop/HajjProject/Ppp", "new_extract_face")
