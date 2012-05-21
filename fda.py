"""
Face detection and augmentation script using OpenCV
"""

import cv


# initialize opencv workings (application window, camera feed)
name    = "tests"
source  = cv.CaptureFromCAM(0)
storage = cv.CreateMemStorage()
cv.NamedWindow(name)

# load all the necessary training sets
facecade  = cv.Load("data/haarcascade_frontalface_alt.xml")
eyecade   = cv.Load("data/haarcascade_eye.xml")
mouthcade = cv.Load("data/haarcascade_mcs_mouth.xml")
learcade  = cv.Load("data/haarcascade_mcs_leftear.xml")
rearcade  = cv.Load("data/haarcascade_mcs_rightear.xml")
nosecade  = cv.Load("data/haarcascade_mcs_nose.xml")

# colors for cv.Rectangle
red    = cv.Scalar(0, 0, 255)
blue   = cv.Scalar(255, 0, 0)
green  = cv.Scalar(0, 128, 0)
yellow = cv.Scalar(255, 255, 0)
purple = cv.Scalar(128, 0, 128)

# bools for which features to detect, set via number keys
face, eyes, mouth, ears, nose = False, False, False, False, False


def detect_feature(frame, cade, color, min_size=(30, 30)):
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
    # which features have what color rectangles
    if face:
        detect_feature(frame, facecade, color=red, min_size=(60, 60))
    if eyes:
        detect_feature(frame, eyecade, color=blue)
    if mouth:
        detect_feature(frame, mouthcade, color=green)
    if ears:
        detect_feature(frame, rearcade, color=yellow)
        detect_feature(frame, learcade, color=yellow)
    if nose:
        detect_feature(frame, nosecade, color=purple)

    # return the processed frame to display
    cv.ShowImage(name, frame)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect_features on each
    frame to look for faces.
    """
    global face, eyes, mouth, ears, nose

    while True:
        # escape key to quit
        if cv.WaitKey(15) == 27:
            break

        # grab key presses to set which features to detect (keys 1-5)
        if cv.WaitKey(15) == 49:
            face = False if face else True
        if cv.WaitKey(15) == 50:
            eyes = False if eyes else True
        if cv.WaitKey(15) == 51:
            mouth = False if mouth else True
        if cv.WaitKey(15) == 52:
            ears = False if ears else True
        if cv.WaitKey(15) == 53:
            nose = False if nose else True

        # grab frame from camera and process
        frame = cv.QueryFrame(source)
        detect_features(frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
