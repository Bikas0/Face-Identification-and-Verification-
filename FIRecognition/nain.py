from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
imageDataset = pickle.load(open('FinalFullPictureDataset.pkl','rb'))
feature_list = np.array(pickle.load(open("FaceEmbedding.pkl",'rb')))
filenames = pickle.load(open('Filenames.pkl','rb'))
detector = MTCNN()


def upload_img(img_path):
    sample_img = cv2.imread(img_path)
    results = detector.detect_faces(sample_img)
    if results == []:
        print("Face not Detected")

    else:
        x, y, width, height = results[0]['box']
        face = sample_img[y:y + height, x:x + width]
        #  extract its features
        image = Image.fromarray(face)
        image = image.resize((224, 224))
        face_array = np.asarray(image)
        face_array = face_array.astype('float32')
        expanded_img = np.expand_dims(face_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img)
        result = model.predict(preprocessed_img).flatten()

        # find the cosine distance of uploded image with all the image feature list
        similarity = []
        for i in range(len(feature_list)):
            similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
        if np.max(similarity) >= 0.70:
            index_positions = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
            positions = []
            for pos in index_positions:
                positions.append(list(pos)[0])

            picture = imageDataset[positions[0]]
            # picture = cv2.resize(picture, (500, 300))
            print("PassPort Number: ", filenames[positions[0]])  # .split('.')[0])
            cv2.imshow(f"{filenames[positions[0]]}", picture)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Unknown")


# call the function
upload_img("Bhaiy.jpeg")
