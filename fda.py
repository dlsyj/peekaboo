"""
The starts of a face detection and augmentation script using OpenCV
"""

import cv


# initialize opencv workings (application window, camera feed)
name = "tests"
source = cv.CaptureFromCAM(0)
cv.NamedWindow(name)
storage = cv.CreateMemStorage()

# load all the necessary training sets
facecade = cv.Load("data/haarcascade_frontalface_alt.xml")
eyecade = cv.Load("data/haarcascade_eye.xml")
mouthcade = cv.Load("data/haarcascade_mcs_mouth.xml")


def detect_feature(frame, cade, color=255, min_size=(60, 60)):
    """detect a single feature and draw a box around the matching area

    By using Canny Pruning, we disregard certain image regions to boost the
    performance. Tweaking the parameters passed to HaarDetectObjects greatly
    affects performance as well.
    """
    # detect feature in image
    objs = cv.HaarDetectObjects(frame, cade, storage,
                                scale_factor=1.2,
                                min_neighbors=3,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                min_size=min_size)
    # draw rectangle around matched feature
    for (x, y, w, h), n in objs:
        cv.Rectangle(frame, (x, y), (x + w, y + h), color)


def detect_features(frame):
    """detect features based on loaded cascades and render the frame"""
    # red, bigger
    detect_feature(frame, facecade,
                   color=cv.Scalar(0, 0, 255), min_size=(60, 60))
    # blue, much smaller
    detect_feature(frame, eyecade,
                   color=cv.Scalar(255, 0, 0), min_size=(30, 30))
    # green, still sort of small
    detect_feature(frame, mouthcade,
                   color=cv.Scalar(0, 128, 0), min_size=(30, 30))

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
        # grab frame from camera and process
        frame = cv.QueryFrame(source)
        detect_features(frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
