import os
import cv2

from constants import normalized_path, emotions_path, images_path, emotions


def list_sorted_dir(path):
    """
    returns sorted list of the directories for the given directory
    :param path: path of the directory
    :return: sorted list of the files/directories
    """
    return sorted(os.listdir(path))


def detect_face(filepath, filename, emotion):
    """
    Detects faces in given files and stores them in normalized data folder.
    As part of the normalization process, it also converts them into grayscale,
    crops the faces and resizes them to 350x350.
    :param filepath: Path of directory where file is stored
    :param filename: Name of the file
    :param emotion: Corresponding emotion as defined in CK+ dataset
    """

    # Initialize the CascadeClassifier for face detection
    # Different cascade configuration files can be used according to
    # the requirements
    face_cascade = cv2.CascadeClassifier(
        '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

    img_path = os.path.join(filepath, filename)
    print("extracting face from %s" % img_path)

    # Read Image object from the file
    img = cv2.imread(img_path)

    # Change the color scheme to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # For each face detected - crop the face, resize it
    # and save it in the normalized folder
    for (x, y, w, h) in faces:
        cropped = img[y:y + h, x:x + w]
        out = cv2.resize(cropped, (350, 350))
        normalized_face_file = os.path.join(normalized_path, emotion, filename)
        print("saving normalized image to %s" % normalized_face_file)
        cv2.imwrite(normalized_face_file, out)


if __name__ == '__main__':

    neutral_prefix = 'neutral'

    # List of all the participants
    participants = list_sorted_dir('%s' % emotions_path)

    for participant in participants:
        participant_path = os.path.join(emotions_path, participant)

        # List of sessions for a participant
        sessions = list_sorted_dir(participant_path)

        # Set neutral done to false initially
        neutral_done = False

        for session in sessions:
            session_path = os.path.join(participant_path, session)

            # List of all the files from the session
            filenames = list_sorted_dir(session_path)

            for filename in filenames:
                # Read the emotion file
                filepath = os.path.join(session_path, filename)
                file = open(filepath)

                # Use the number to get the index from the emotion list
                emotion_idx = int(float(file.readline()))

                # Read the image file
                img_path = os.path.join(images_path, participant, session)

                # List of all the images in the image directory
                src_img_list = list_sorted_dir(img_path)

                # Take the first image as neutral and the last image as
                # the full expression of the emotion
                src_first_img = list_sorted_dir(img_path)[0]
                src_final_img = list_sorted_dir(img_path)[-1]

                # Only one neutral image per subject
                if not neutral_done:
                    detect_face(img_path, src_first_img, neutral_prefix)
                    neutral_done = True

                # Call detect_face to normalize the image
                detect_face(img_path, src_final_img, emotions[emotion_idx])
