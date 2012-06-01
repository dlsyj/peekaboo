"""
Derek Hurley
6/6/12
CS 510
Term Project

Facial Feature Detection and Augmentation
"""

import cv


# load all the necessary training sets
face_cade  = cv.Load("data/haarcascade_frontalface_alt.xml")
eye_cade   = cv.Load("data/haarcascade_eye.xml")
nose_cade  = cv.Load("data/haarcascade_mcs_nose.xml")

# images to overlay on features
moustache = cv.LoadImage("data/moustache.png")
eyeball   = cv.LoadImage("data/eye.png")
tophat    = cv.LoadImage("data/tophat.png")

# drawn bottom up rather than top down
cv.Flip(tophat, tophat, flipMode=0)

# bools for which features to detect, set via number keys
face_on, eye_on, nose_on = False, False, False


def overlay_image(frame, image, x, y, w, h):
    """resize and overlay an image where a feature is detected

    This resizes the corresponding image to a matched feature, then loops
    through all of its pixels to superimpose the image on the frame
    """
    # resize the image to fit the detected feature
    new_feature = cv.CreateImage((w, h), 8, 3)
    cv.Resize(image, new_feature, interpolation=cv.CV_INTER_AREA)

    # overlay the image on the frame
    for py in xrange(h):
        for px in xrange(w):
            pixel = cv.Get2D(new_feature, py, px)

            # don't map the whitespace surrounding the image
            if pixel != (255.0, 255.0, 255.0, 0.0):
                if image is tophat:
                    # above feature
                    new_y = y - py
                elif image is moustache:
                    # bottom half of feature
                    new_y = (h / 2) + y + py
                else:
                    # over feature
                    new_y = y + py
                new_x = x + px

                # make sure the image is in the frame
                if 0 < new_x < frame.width and 0 < new_y < frame.height:
                    cv.Set2D(frame, new_y, new_x, pixel)


def detect_feature(frame, cade, min_size=(30, 30)):
    """detect a single feature and draw a box around the matching area

    By using Canny Pruning, we disregard certain image regions to boost the
    performance. Tweaking the parameters passed to HaarDetectObjects greatly
    affects performance as well.
    """
    # detect feature in image
    objs = cv.HaarDetectObjects(frame, cade, cv.CreateMemStorage(),
                                scale_factor=1.2, min_neighbors=3,
                                flags=cv.CV_HAAR_DO_CANNY_PRUNING,
                                min_size=min_size)

    # pass the proper image to overlay for each match
    for (x, y, w, h), n in objs:
        if cade == face_cade:
            overlay_image(frame, tophat, x, y, w, h)
        elif cade == eye_cade:
            overlay_image(frame, eyeball, x, y, w, h)
        elif cade == nose_cade:
            overlay_image(frame, moustache, x, y, w, h)


def detect_features(frame):
    """detect features based on loaded cascades and render the frame"""
    if face_on:
        detect_feature(frame, face_cade, min_size=(60, 60))
    if eye_on:
        detect_feature(frame, eye_cade)
    if nose_on:
        detect_feature(frame, nose_cade)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect_features on each
    frame to look for True features (based on keys 1-3).
    """
    global face_on, eye_on, nose_on

    # set up window and camera
    win_name = "peekaboo"
    source = cv.CaptureFromCAM(0)
    cv.NamedWindow(win_name)

    while True:
        # grab the key press to delegate action
        key = cv.WaitKey(15)

        # escape key to exit
        if key == 27:
            break

        # keys 1-3 determine detected features
        if key == 49:
            face_on = False if face_on else True
        if key == 50:
            eye_on = False if eye_on else True
        if key == 51:
            nose_on = False if nose_on else True

        # get the frame, detect selected features, and show frame
        frame = cv.QueryFrame(source)
        detect_features(frame)
        cv.ShowImage(win_name, frame)

    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
