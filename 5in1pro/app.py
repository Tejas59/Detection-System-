from flask import Flask, render_template, request, url_for, redirect, flash, Response, session
#from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from flask_bcrypt import Bcrypt
import cv2
import os
import imutils
import numpy as np
import copy
from tensorflow.keras.models import load_model
import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)
bcrypt = Bcrypt(app)


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    faces = []
    locs = []
    preds = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# load our serialized face detector model from disk
prototxtPath = 'F:/Project2_final/5in1pro/face_detector/deploy.prototxt'
weightsPath = 'F:/Project2_final/5in1pro/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("F:/Project2_final/5in1pro/mask_detector.model")


def mask_frames():
    cap = cv2.VideoCapture(0)

    while True:
        flag, frame = cap.read()
        frame = imutils.resize(frame, width=800)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
            if mask > withoutMask:
                label = "Mask"
                color = (0, 255, 0)
            else:
                label = "No Mask"
                color = (0, 0, 255)
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# motion detection


def motion_detection():
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

    out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

    ret, frame1 = cap.read()
    ret, frame2 = cap.read()
    # print(frame1.shape)
    while cap.isOpened():
        diff = cv2.absdiff(frame1, frame2)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)

            if cv2.contourArea(contour) < 900:
                continue
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame1)
        frame1 = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n\r\n')
        frame1 = frame2
        ret, frame2 = cap.read()

# emotion detection


def emotion_detection():
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),
              activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    model.load_weights('F:/Project2_final/5in1pro/model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                    3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # start the webcam feed
    cap = cv2.VideoCapture(0)
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), thickness=7)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(
                cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# height detection
def height_detection():
    def non_max_suppression(boxes, overlapThresh):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return boxes[pick].astype("int")

    _offset = 50
    dim_reference_obj_size = 8  # in cm
    logo_height = 34

    reference_scale = 0.3
    reference_logo = cv2.imread(os.path.join("F:/Project2_final/5in1pro/height/images/try3.jpg"), 0)
    reference_logo = cv2.resize(reference_logo, (int(
        reference_logo.shape[1] * reference_scale), int(reference_logo.shape[0] * reference_scale)))
    reference_logo = cv2.Canny(reference_logo, 100, 200)
    tH, tW = reference_logo.shape[:2]

    camera = cv2.VideoCapture(0)
    # HOG for human detection
    hog_descriptor = cv2.HOGDescriptor()
    hog_descriptor.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # HAAR cascades for face detection
    face_haarcascade = cv2.CascadeClassifier(os.path.join(
        cv2.data.haarcascades+"haarcascade_frontalface_default.xml"))
    scales = np.linspace(0.02, 1.0, 20)[::-1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    while (True):
        isFrameReadCorrect, frame = camera.read()
        if isFrameReadCorrect:
            frame = cv2.resize(
                frame, (int(frame.shape[1] * 0.80), int(frame.shape[0] * 0.80)))
            frame_tracked = copy.deepcopy(frame)

            # gray = cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2GRAY)
            found = None
            gray = cv2.cvtColor(frame_tracked, cv2.COLOR_BGR2GRAY)
            for scale in scales:
                resized = cv2.resize(
                    gray, (int(gray.shape[1] * scale), int(gray.shape[0] * scale)))
                if resized.shape[0] < tH or resized.shape[1] < tW:
                    break

                rH = gray.shape[1] / float(resized.shape[1])
                rW = gray.shape[0] / float(resized.shape[0])
                edged = cv2.Canny(resized, 100, 200)
                result = cv2.matchTemplate(
                    edged, reference_logo, cv2.TM_CCOEFF)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, rH, rW)
            logo = None
            logo_dim = None
            if not found is None and found[0] > 5500000.0:
                (_, maxLoc, rH, rW) = found
                (startX, startY) = (int(maxLoc[0] * rW), int(maxLoc[1] * rH))
                (endX, endY) = (
                    int((maxLoc[0] + tW) * rW), int((maxLoc[1] + tH) * rH))
                logo = ((startX+endX)/2, (startY+endY)/2)
                logo_dim = (max(0, startX-_offset), min(endX+_offset,
                            frame_tracked.shape[1]-1), startY, endY)
                cv2.rectangle(frame, (startX, startY),
                              (endX, endY), (0, 255, 0), 2)

            if logo_dim != None:
                logo_ratio = logo_height / abs(logo_dim[3]-logo_dim[2])
                (rects, weights) = hog_descriptor.detectMultiScale(
                    frame_tracked, winStride=(4, 3), padding=(0, 0), scale=1.06)
                rects = np.array([[x, y, x + w, y + h]
                                 for (x, y, w, h) in rects])
                pick = non_max_suppression(rects, overlapThresh=0.60)
                if (pick is not None and len(pick) > 0):
                    for (xA, yA, xB, yB) in pick:
                        human_frame = gray[yA:yB, xA:xB]
                        faces = face_haarcascade.detectMultiScale(
                            human_frame, 1.3, 5)
                        if not faces is None and len(faces) == 1:
                            (x, y, face_width, face_height) = faces[0]
                            # cv2.rectangle(frame, (xA+x, yA+y), (xA+x + face_width, yA+y + face_height), (0, 255, 237), 3)
                            yA = max(yA, yA+y)
                        # estimate human height
                        human_height = (yB-yA) * logo_ratio
                        human_height_ft = int(human_height / 9)
                        human_height_inch = int(human_height % 9) / 2.54
                        cv2.rectangle(frame, (xA, yA),
                                      (xB, yB), (0, 255, 0), 2)
                        cv2.putText(frame, str(human_height_ft)+" feet " + "{0:.0f}".format(
                            human_height_inch)+" inches", (xA+10, yA-10), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
                        #print(str(human_height_ft)+" feet "+ "{0:.0f}".format(human_height_inch)+" inches",)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


# height_detection()

# name detection

def name_detection():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('F:/Project2_final/5in1pro/name/trainer/trainer.yml')
    cascadePath = 'F:/Project2_final/5in1pro/name/haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'shrutisha']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video widht
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1*cam.get(3)
    minH = 0.1*cam.get(4)

    while True:

        ret, frame = cam.read()
        # img = cv2.flip(img, -1) # Flip vertically

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for(x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Check if confidence is less them 100 ==> "0" is perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(frame, str(id), (x+5, y-5),
                        font, 1, (255, 255, 255), 2)
            #cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

        # cv2.imshow('camera',img)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

# face_detectore()


@app.route('/name_video', methods=['GET', 'POST'])
def name_video():
    return Response(name_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/name', methods=['GET', 'POST'])
def name():
    return render_template('name.html')


@app.route('/dimension', methods=['GET', 'POST'])
def dimension():
    return render_template('dimension.html')


@app.route('/dimension_video', methods=['GET', 'POST'])
def dimension_video():
    return Response(height_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/', methods=['GET', 'POST'])
def first():
    return redirect(url_for('start'))


@app.route('/mask', methods=['GET'])
def mask():
    return render_template('mask.html')


@app.route('/mask_video', methods=['GET'])
def mask_video():
    return Response(mask_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/motion', methods=['GET'])
def motion():

    return render_template('motion.html')


@app.route('/motion_video', methods=['GET'])
def motion_video():
    return Response(motion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/emotion', methods=['GET'])
def emotion():

    return render_template('emotion.html')


@app.route('/emotion_video', methods=['GET'])
def emotion_video():
    return Response(emotion_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/start", methods=["GET", "POST"])
def start():
    return render_template("start.html")


if __name__ == '__main__':
    app.run(debug=True)
