import numpy as np
from PIL import Image
from mtcnn import MTCNN
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
import cv2
import pickle
import os
from sklearn.metrics.pairwise import cosine_similarity
import time
start = time.time()
model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = np.array(pickle.load(open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join("pklfile", 'FinalFilenames.pkl'),'rb'))

detector = MTCNN()

def upload_img(img_path):
    sample_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(sample_img)
    sample_img = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
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

        similarity = [cosine_similarity(result.reshape(1, -1), feature.reshape(1, -1))[0][0] for feature in
                      feature_list]

        # Create a list of (filename, similarity) pairs
        filename_similarity_pairs = list(zip(filenames, similarity))

        # Sort the list based on similarity scores in descending order
        sorted_pairs = sorted(filename_similarity_pairs, key=lambda x: x[1], reverse=True)

        # Print filenames with similarity scores above 0.50
        for filename, score in sorted_pairs:
            if score >= 0.70:
                print("Similarity Score:", score, "Passport Number:", filename)
            else:
                break  # Stop printing when similarity drops below 0.50
        else:
            print("Unknown")


# call the function
upload_img("mahedi.jpg")
end = time.time()
print("Time: ", end-start)
#
#

# import numpy as np
# from PIL import Image
# from mtcnn import MTCNN
# from keras_vggface.utils import preprocess_input
# from keras_vggface.vggface import VGGFace
# import cv2
# import pickle
# from tqdm import tqdm
# import os
# from sklearn.metrics.pairwise import cosine_similarity
#
# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# imageDataset = pickle.load(open(os.path.join("pklfile",'FinalFullPictureDataset.pkl'),'rb'))
# feature_list = np.array(pickle.load(open(os.path.join("pklfile",'FeatureEmbeddings.pkl'),'rb')))
# filenames = pickle.load(open(os.path.join("pklfile",'FinalFilenames.pkl'),'rb'))
# detector = MTCNN()
#
# count = 0
#
#
# def upload_img(img_path):
#     img_path = os.path.join('hajj_images',img_path)
#     sample_img = cv2.imread(img_path)
#     results = detector.detect_faces(sample_img)
#     if results == []:
#         print("Face not Detected")
#
#     else:
#         x, y, width, height = results[0]['box']
#         face = sample_img[y:y + height, x:x + width]
#         #  extract its features
#         image = Image.fromarray(face)
#         image = image.resize((224, 224))
#         face_array = np.asarray(image)
#         face_array = face_array.astype('float32')
#         expanded_img = np.expand_dims(face_array, axis=0)
#         preprocessed_img = preprocess_input(expanded_img)
#         result = model.predict(preprocessed_img).flatten()
#
#
#
#         # find the cosine distance of uploded image with all the image feature list
#         similarity = []
#         for i in range(len(feature_list)):
#             similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
#         if np.max(similarity) >= 0.70:
#             index_positions = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
#             positions = []
#             for pos in index_positions:
#                 positions.append(list(pos)[0])
#
#             # picture = imageDataset[positions[0]]
#             # picture = cv2.resize(picture, (500, 300))
#             print("PassPort Number: ", filenames[positions[0]]+"   "+img_path.split('\\')[1])
#             if filenames[positions[0]]!=img_path.split('\\')[1]:
#                 global count
#                 count = count +1
#                 print("Wrong: ", count)
#             # .split('.')[0])
#             # cv2.imshow(f"{filenames[positions[0]]}", picture)
#             # cv2.waitKey(0)
#             # cv2.destroyAllWindows()
#         else:
#             print("Unknown")
#
#
# # call the function
# # upload_img("hajj_images/N2A0BB4BB44.jpg")
#
# for file in tqdm(os.listdir("hajj_images")):
#     print(file)
#     upload_img(file)
