from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2
import keras
import os
from keras.preprocessing import image
import pickle
import numpy as np

person_file_names = os.listdir('FinalFolder')
filenames = []
for person_file_name in person_file_names:
    filenames.append(person_file_name)
pickle.dump(filenames, open('Filenames.pkl', 'wb'))


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')


def feature_extractor(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)
    result = model.predict(preprocessed_img).flatten()
    return result


features = []
for file in filenames:
    features.append(feature_extractor(os.path.join("NewTestFaceExtract", file), model))
pickle.dump(features, open('FaceEmbedding.pkl', 'wb'))


# List to store images
images = []
for path in os.listdir("FinalFolder"):
    img = cv2.imread(os.path.join("FinalFolder", path))
    images.append(img)
pickle.dump(images, open('FinalFullPictureDataset.pkl', 'wb'))