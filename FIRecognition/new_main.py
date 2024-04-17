# import os
# import cv2
# import pickle
# import numpy as np
# from PIL import Image
# from mtcnn import MTCNN
# from numpy import asarray
# from tqdm import tqdm
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
# from tensorflow.keras.preprocessing import image
# from sklearn.metrics.pairwise import cosine_similarity
#
#
# model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
# feature_list = np.array(pickle.load(open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'rb')))
# filenames = pickle.load(open(os.path.join("pklfile", 'FinalFilenames.pkl'), 'rb'))
#
#
# def delete_files_in_folder(folder_path):
#     try:
#         files = os.listdir(folder_path)
#         for file_name in files:
#             file_path = os.path.join(folder_path, file_name)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)
#     except Exception as e:
#         print(f"An error occurred: {e}")
#
#
# def feature_extractor(extract_face_dir, img_path, model):
#     img = image.load_img(os.path.join(extract_face_dir, img_path), target_size=(224,224))
#     img = cv2.cvtColor(img, )
#     img_array = image.img_to_array(img)
#     expanded_img = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#
#     return result
#
#
# count = 0
#
#
# def extract_faces_mtcnn(img_path, output_folder, required_size=(224, 224)):
    # Create the MTCNN face detector
#     detector = MTCNN()
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#     # Loop through each file in the input folder
#
#     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     # Apply CLAHE
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_image = clahe.apply(img)
#     ConvertedImage = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
#     # Detect faces in the image using MTCNN
#     faces = detector.detect_faces(ConvertedImage)
#     if faces == []:
#         pass
#         # print("Face Not Detected")
#     else:
#         # Loop through each detected face
#         for i, face_info in enumerate(faces):
#             x, y, w, h = face_info['box']
#             x, y = max(x, 0), max(y, 0)  # Ensure coordinates are non-negative
#             face = img[y:y + h, x:x + w]
#             # resize pixels to the model size
#             image = Image.fromarray(face)
#             image = image.resize(required_size)
#             face_array = asarray(image)
#             face_filename = f"{img_path}"
#             # print(face_filename)
#             # Save the face in the output folder
#             output_path = os.path.join(output_folder, face_filename)
#             cv2.imwrite(output_path, face_array)
#     # print("Face extraction complete.")
#
#
# def upload_img(img_file_path):
#     extract_face_dir = "verification_img_dir"
#     # Call the function to extract faces using MTCNN
#     extract_faces_mtcnn(img_file_path, extract_face_dir)
#     if len(os.listdir(extract_face_dir)) == 0:
#         print("Face Not Detected")
#     else:
#         result = feature_extractor(extract_face_dir, img_file_path, model)
#
#         # # find the cosine distance of uploaded image with all the image feature list
#         # similarity = cosine_similarity(result.reshape(1, -1), feature_list)
#         # max_similarity = np.max(similarity)
#         # if max_similarity >= 0.60:
#         #     index = np.argmax(similarity)
#         #     # print("Passport Number:", filenames[index])
#         #     print(f"PassPort Number: {filenames[index]} == {img_file_path}")
#         #     if filenames[index] != img_file_path:  # .split('\\')[1]:
#         #         global count
#         #         count = count + 1
#         #         print("Wrong: ", count)
#         # else:
#         #     print("Unknown")
#
#         # # find the cosine distance of uploded image with all the image feature list
#         similarity = []
#         for i in range(len(feature_list)):
#             similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
#         if np.max(similarity) >= 0.60:
#             index_positions = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
#             positions = []
#             for pos in index_positions:
#                 positions.append(list(pos)[0])
#             print("PassPort Number: ", filenames[positions[0]])
#             if filenames[positions[0]] != img_file_path.split('\\')[1]:
#                 global count
#                 count = count +1
#                 print("Wrong: ", count)
#             # .split('.')[0])
#         else:
#             print("Unknown")
#
#         delete_files_in_folder(extract_face_dir)
#
#
# upload_img("16.jpg")


import os
import cv2
import pickle
import numpy as np
from PIL import Image
from mtcnn import MTCNN
from numpy import asarray
from tqdm import tqdm
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
feature_list = np.array(pickle.load(open(os.path.join("pklfile", 'FeatureEmbeddings.pkl'), 'rb')))
filenames = pickle.load(open(os.path.join("pklfile", 'FinalFilenames.pkl'), 'rb'))


def delete_files_in_folder(folder_path):
    try:
        files = os.listdir(folder_path)
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)
    except Exception as e:
        print(f"An error occurred: {e}")

def feature_extractor(extract_face_dir, img_path, model):
    # Load and convert the image to RGB (if not already in RGB format)
    img = cv2.imread(os.path.join(extract_face_dir, img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))

    # Convert the image to array and preprocess
    img_array = image.img_to_array(img)
    expanded_img = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img)

    # Extract features using the model
    result = model.predict(preprocessed_img).flatten()

    return result
# def feature_extractor(extract_face_dir, img_path, model):
#     img = image.load_img(os.path.join(extract_face_dir, img_path), target_size=(224,224))
#     img_array = image.img_to_array(img)
#     expanded_img = np.expand_dims(img_array, axis=0)
#     preprocessed_img = preprocess_input(expanded_img)
#     result = model.predict(preprocessed_img).flatten()
#
#     return result


count = 0
face_not_detected = 0


def extract_faces_mtcnn(img_path, output_folder, required_size=(224, 224)):
    # Create the MTCNN face detector
    detector = MTCNN()
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Loop through each file in the input folder

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(img)
    ConvertedImage = cv2.cvtColor(clahe_image, cv2.COLOR_GRAY2BGR)
    # Detect faces in the image using MTCNN
    faces = detector.detect_faces(ConvertedImage)
    if faces == []:
        pass
        # print("Face Not Detected")
    else:
        # Loop through each detected face
        for i, face_info in enumerate(faces):
            x, y, w, h = face_info['box']
            x, y = max(x, 0), max(y, 0)  # Ensure coordinates are non-negative
            face = img[y:y + h, x:x + w]
            # resize pixels to the model size
            image = Image.fromarray(face)
            image = image.resize(required_size)
            face_array = asarray(image)
            face_filename = f"{img_path}"
            # face_filename = face_filename.split("\\")[1]
            # print(face_filename)
            # Save the face in the output folder
            output_path = os.path.join(output_folder, face_filename)
            cv2.imwrite(output_path, face_array)
    # print("Face extraction complete.")


def upload_img(img_file_path):
    extract_face_dir = "verification_img_dir"
    # img_file_path = os.path.join('hajj_images', img_file_path)
    # Call the function to extract faces using MTCNN
    extract_faces_mtcnn(img_file_path, extract_face_dir)
    if len(os.listdir(extract_face_dir)) == 0:
        print("Face Not Detected")
        x = 0
        if x == 0:  # .split('\\')[1]:
            global face_not_detected
            face_not_detected = face_not_detected + 1
            print("Face_not_detected: ", face_not_detected)

    else:
        # img_file_path = img_file_path.split("\\")[1]
        result = feature_extractor(extract_face_dir, img_file_path, model)

        # find the cosine distance of uploaded image with all the image feature list
        similarity = cosine_similarity(result.reshape(1, -1), feature_list)
        max_similarity = np.max(similarity)
        print(max_similarity)
        if max_similarity >= 0.60:
            index = np.argmax(similarity)
            # print("Passport Number:", filenames[index])
            print(f"PassPort Number: {filenames[index]} == {img_file_path}")
            if filenames[index] != img_file_path:  # .split('\\')[1]:
                global count
                count = count + 1
                print("Wrong: ", count)
        else:
            print("Unknown")


        # # find the cosine distance of uploded image with all the image feature list
        # similarity = []
        # for i in range(len(feature_list)):
        #     similarity.append(cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
        # if np.max(similarity) >= 0.70:
        #     index_positions = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])
        #     positions = []
        #     for pos in index_positions:
        #         positions.append(list(pos)[0])
        #     print(f"PassPort Number: {filenames[positions[0]]} == {img_file_path}")
        #     if filenames[positions[0]] != img_file_path: #.split('\\')[1]:
        #         global count
        #         count = count + 1
        #         print("Wrong: ", count)
        #     # .split('.')[0])
        # else:
        #     print("Unknown")

        # delete_files_in_folder(extract_face_dir)


upload_img("FinalFolder/H7F94C53C1.jpeg")
# for file in tqdm(os.listdir("hajj_images")):
#     print("File Name", file)
#     upload_img(file)
#     print("\n")

