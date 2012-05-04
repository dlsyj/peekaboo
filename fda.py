"""
The starts of a face detection and augmentation script using OpenCV
"""

import cv


# initialize opencv workings (application window, camera feed)
name = "tests"
source = cv.CaptureFromCAM(0)
cv.NamedWindow(name)

# load all the necessary training sets
facecade = cv.Load("data/haarcascade_frontalface_default.xml")
eyecade = cv.Load("data/haarcascade_eye.xml")
mouthcade = cv.Load("data/haarcascade_mcs_mouth.xml")

# consistent HaarDetectObjects parameters
scale = 1.2
flags = cv.CV_HAAR_DO_CANNY_PRUNING
neighbors = 2

# storage for opencv work
storage = cv.CreateMemStorage()


def detect_feature(frame, cade, color=255):
    """detect a single feature and draw a box around the matching area

    By using Canny Pruning, we disregard certain image regions to boost the
    performance. Tweaking the parameters passed to HaarDetectObjects greatly
    affects performance as well.
    """
    # detect feature in image
    objs = cv.HaarDetectObjects(frame, cade, storage, scale_factor=scale,
                                min_neighbors=neighbors, flags=flags,
                                min_size=(80, 80))
    # draw rectangle around matched feature
    for (x, y, w, h), n in objs:
        cv.Rectangle(frame, (x, y), (x + w, y + h), color)


def detect_features(frame):
    """detect features based on loaded cascades and render the frame"""
    detect_feature(frame, facecade, color=cv.Scalar(0, 0, 255))
    detect_feature(frame, eyecade, color=cv.Scalar(255, 0, 0))
    detect_feature(frame, mouthcade, color=cv.Scalar(0, 128, 0))

    # return the processed frame to display
    cv.ShowImage(name, frame)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect_features on each
    frame to look for faces.
    """
    while True:
        # escape key to quit
        if cv.WaitKey(15) == 27:
            break
        # grab frame from camera and
        frame = cv.QueryFrame(source)
        detect_features(frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
