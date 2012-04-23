"""
The starts of a face detection and augmentation script using OpenCV
"""

import cv


# initialize opencv workings (haar cascades, camera feed)
name = "tests"
source = cv.CaptureFromCAM(0)    # video source (iSight is indexed at 0)
cv.NamedWindow(name)
facecade = cv.Load("data/haarcascade_frontalface_default.xml")
storage = cv.CreateMemStorage()    # storage for opencv work


def detect(frame):
    """detect the facial features and draw rectangles around matches

    By using Canny Pruning, we disregard certain image regions to boost the
    detection performance. Note that face detection is only supported.
    """
    objs = cv.HaarDetectObjects(frame,
                                facecade,
                                storage,
                                min_neighbors=2,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING)
    for (x, y, w, h), n in objs:
        cv.Rectangle(frame, (x, y), (x + w, y + h), 255)
    cv.ShowImage(name, frame)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect() on each frame to
    look for faces. On breaking from the loop, the windows are destroyed.
    """
    while True:
        if cv.WaitKey(15) == 27:    # escape key to quit
            break
        frame = cv.QueryFrame(source)    # grab frame from camera and
        detect(frame)

    cv.DestroyAllWindows()    # close GUI Window


if __name__ == '__main__':
    loop()
