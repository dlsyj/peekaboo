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

# storage for opencv work
storage = cv.CreateMemStorage()


def detect(frame):
    """detect the facial features and draw rectangles around matches

    By using Canny Pruning, we disregard certain image regions to boost the
    detection performance. Note that face detection is only supported. Tweaking
    the parameters passed to HaarDetectObjects greatly affects performance.
    """
    # detect the entire face
    fobjs = cv.HaarDetectObjects(frame,
                                 facecade,
                                 storage,
                                 scale_factor=1.2,
                                 min_neighbors=2,
                                 flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                 min_size=(80, 80))
    # draw facial rectangle
    for (x, y, w, h), n in fobjs:
        blue = cv.Scalar(0, 0, 255)
        cv.Rectangle(frame, (x, y), (x + w, y + h), blue)

    # detect eyes
    eobjs = cv.HaarDetectObjects(frame,
                                 eyecade,
                                 storage,
                                 scale_factor=1.2,
                                 min_neighbors=2,
                                 flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                 min_size=(30, 30))
    # draw rectangle around eyes
    for (x, y, w, h), n in eobjs:
        red = cv.Scalar(255, 0, 0)
        cv.Rectangle(frame, (x, y), (x + w, y + h), red)

    # detect mouth
    mobjs = cv.HaarDetectObjects(frame,
                                 mouthcade,
                                 storage,
                                 scale_factor=1.2,
                                 min_neighbors=2,
                                 flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                 min_size=(30, 30))
    # draw rectangle around mouth
    for (x, y, w, h), n in mobjs:
        green = cv.Scalar(0, 128, 0)
        cv.Rectangle(frame, (x, y), (x + w, y + h), green)

    # return the processed frame to display
    cv.ShowImage(name, frame)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect() on each frame to
    look for faces. On breaking from the loop, the windows are destroyed.
    """
    while True:
        # escape key to quit
        if cv.WaitKey(15) == 27:
            break
        # grab frame from camera and
        frame = cv.QueryFrame(source)
        detect(frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
