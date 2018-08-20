import os
import random
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from constants import normalized_path

# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions = ["neutral", "anger", "disgust", "happy", "surprise"]

def load_data(split=True):
    """
    Loads images from the normalized directory, splits them into
    training (80%) and testing sets (20%).
    :return:
    """

    # Training set containing images
    _X_train = []

    # Test set containing images
    _X_test = []

    # Training set containing labels corresponding to training images
    _y_train = []

    # Test set containing labels corresponding to test images
    _y_test = []

    for emotion in emotions:
        img_path = os.path.join(normalized_path, emotion)
        images = os.listdir(img_path)

        # Shuffle the images
        random.shuffle(images)
        if split:
            train, test = train_test_split(images, test_size=0.2)
        else:
            train = images
            test = []

        for img_train in train:
            # Read it in grayscale mode
            gray_img_train = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_train)), cv2.COLOR_BGR2GRAY)

            # Add the image to training set
            _X_train.append(gray_img_train)

            # Add the label to training set
            _y_train.append(emotions.index(emotion))

        for img_test in test:
            # Read it in grayscale mode
            gray_img_test = cv2.cvtColor(cv2.imread(os.path.join(img_path, img_test)), cv2.COLOR_BGR2GRAY)

            # Add the image to test set
            _X_test.append(gray_img_test)

            # Add the label to test set
            _y_test.append(emotions.index(emotion))

    return _X_train, _X_test, _y_train, _y_test


def test_model(model):
    haar_default_clf = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml")
    haar_alt2_clf = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml")
    haar_alt_clf = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")
    haar_alt_tree_clf = cv2.CascadeClassifier(
        "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt_tree.xml")

    img = cv2.imread('testimages/1.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_haar_default = haar_default_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                          flags=cv2.CASCADE_SCALE_IMAGE)
    face_haar_alt2 = haar_alt2_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
    face_haar_alt = haar_alt_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
    face_haar_alt_tree = haar_alt_tree_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                                            flags=cv2.CASCADE_SCALE_IMAGE)

    if len(face_haar_default) == 1:
        faces = face_haar_default
    elif len(face_haar_alt2) == 1:
        faces = face_haar_alt2
    elif len(face_haar_alt) == 1:
        faces = face_haar_alt
    elif len(face_haar_alt_tree) == 1:
        faces = face_haar_alt_tree
    else:
        faces = ""

    print(len(faces))
    for (x, y, w, h) in faces:
        gray = gray[y:y + h, x:x + w]  # Cut the frame to size
        try:
            out = cv2.resize(gray, (350, 350))
            cv2.imwrite('testimages/1_2.jpg', out)
        except:
            pass

    im = cv2.imread('testimages/1_2.jpg')
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    em, c = model.predict(g)
    assert emotions[em] == 'happy'


def train_predict(model):
    model.train(X_train, np.asarray(y_train))

    print("Predicting...")
    idx = 0
    positive = 0
    negative = 0

    for test_img in X_test:
        pred, conf = model.predict(test_img)
        if pred == y_test[idx]:
            positive += 1
        else:
            negative += 1
        idx += 1

    # Accuracy of the model:
    score = (100 * positive) / (positive + negative)
    print(score)
    return model, score



if __name__ == '__main__':
    # Load data from the normalized set
    X_train, X_test, y_train, y_test = load_data(True)
    print(len(X_train))
    print("Training Fisherface Recognizer...")

    fisherface_recognizer = cv2.face.FisherFaceRecognizer_create()

    scores = {}

    fisherface_recognizer, scores['fisherface'] = train_predict(fisherface_recognizer)

    print(scores)

    # Test the model with an external image
    test_model(fisherface_recognizer)

    # X_train, X_test, y_train, y_test = load_data(False)
    # print(len(X_train))
    #
    # # Train the model again with complete dataset:
    # fisherface_recognizer = cv2.face.FisherFaceRecognizer_create()
    # fisherface_recognizer.train(X_train, np.asarray(y_train))

    # Save the final model to an XML file
    fisherface_recognizer.write("model2.xml")