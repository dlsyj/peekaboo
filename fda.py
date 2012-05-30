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

# first image overlay
face_image = cv.LoadImage("data/poker_face.png")

# bools for which features to detect, set via number keys
face, eyes, mouth, ears, nose = False, False, False, False, False


def overlay_image(frame, x, y, w, h):
    """resize and overlay an image where a feature is detected

    This resizes the corresponding image to a matched feature, then loops
    through all of its pixels to superimpose the image on the frame
    """
    global face_image

    new_feature = cv.CreateImage((w, h), 8, 3)
    cv.Resize(face_image, new_feature, interpolation=cv.CV_INTER_AREA)

    # overlay the face_image on the frame
    for px in xrange(w):
        for py in xrange(h):
            over = cv.Get2D(new_feature, py, px)
            # this map takes care of the transparency of the image
            over_ready = tuple(map(lambda x: x + 1, over))
            cv.Set2D(frame, py + y, px + x, over_ready)


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
        overlay_image(frame, x, y, w, h)


def detect_features(frame):
    """detect features based on loaded cascades and render the frame"""
    # colors for cv.Rectangle
    red    = cv.Scalar(0, 0, 255)
    blue   = cv.Scalar(255, 0, 0)
    green  = cv.Scalar(0, 128, 0)
    yellow = cv.Scalar(255, 255, 0)
    purple = cv.Scalar(128, 0, 128)

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
        # grab the key press to delegate action
        key = cv.WaitKey(15)

        # escape key
        if key == 27:
            break

        # keys 1-5 determine detected features
        if key == 49:
            face = False if face else True
        if key == 50:
            eyes = False if eyes else True
        if key == 51:
            mouth = False if mouth else True
        if key == 52:
            ears = False if ears else True
        if key == 53:
            nose = False if nose else True

        # grab frame from camera and process
        frame = cv.QueryFrame(source)
        detect_features(frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
