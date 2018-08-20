import os

import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

UPLOAD_FOLDER = '/home/ec2-user/code/emotag/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# emotions = ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]
emotions = ["neutral", "angry", "disgusted", "happy", "surprised"]


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Function to resize the image to resolution lesser than 800x600
    while maintaining the aspect ratio.
    :param image: image object
    :param width: desired width of the image
    :param height: desired height of the image
    :param inter: interpolation
    :return:
    """

    # initialize the dimensions of the image to be resized and
    # grab the image size
    dimension = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dimension = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dimension = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dimension, interpolation=inter)

    # return the resized image
    return resized


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(f)

    haar_alt_clf = cv2.CascadeClassifier("/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml")

    img = cv2.imread(f)
    img = image_resize(img, width=800)
    (h, w) = img.shape[:2]
    if not h <= 600:
        img = image_resize(img, height=600)

    cv2.imwrite(f, img)

    img = cv2.imread(f)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_faces = haar_alt_clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                               flags=cv2.CASCADE_SCALE_IMAGE)

    fisherface_recognizer = cv2.face.FisherFaceRecognizer_create()
    fisherface_recognizer.read("model2.xml")

    data = {}
    faces = []
    for (x, y, w, h) in haar_faces:
        gray_face = gray[y:y + h, x:x + w]  # Cut the frame to size
        try:
            out = cv2.resize(gray_face, (350, 350))
            em, c = fisherface_recognizer.predict(out)
            face_rectangle = {'top': int(x), 'left': int(y), 'width': int(w), 'height': int(h), 'emotion': emotions[em]}
            faces.append(face_rectangle)
        except:
            pass

    data['faces'] = faces
    data['img'] = "uploads/" + file.filename
    height, width, channels = img.shape
    data['imgsize'] = {'height': height, 'width': width}

    return jsonify(data)


@app.route('/uploads/<path:path>')
def send_picture(path):
    return send_from_directory('uploads', path)


if __name__ == '__main__':
    app.run("0.0.0.0", "5432")
