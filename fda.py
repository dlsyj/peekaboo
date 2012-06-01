"""
Face detection and augmentation script using OpenCV
"""

import cv


# load all the necessary training sets
face_cade  = cv.Load("data/haarcascade_frontalface_alt.xml")
eye_cade   = cv.Load("data/haarcascade_eye.xml")
nose_cade  = cv.Load("data/haarcascade_mcs_nose.xml")

# first image overlay
moustache = cv.LoadImage("data/moustache.png")
eyeball   = cv.LoadImage("data/eye.png")
tophat    = cv.LoadImage("data/tophat.png")

# drawn bottom up rather than top down
cv.Flip(tophat, tophat, flipMode=0)

# bools for which features to detect, set via number keys
face, eye, nose = False, False, False


def overlay_image(frame, image, x, y, w, h):
    """resize and overlay an image where a feature is detected

    This resizes the corresponding image to a matched feature, then loops
    through all of its pixels to superimpose the image on the frame
    """
    new_feature = cv.CreateImage((w, h), 8, 3)
    cv.Resize(image, new_feature, interpolation=cv.CV_INTER_AREA)

    # overlay the image on the frame
    for py in xrange(h):
        for px in xrange(w):
            over = cv.Get2D(new_feature, py, px)

            # don't map the whitespace surrounding the image
            if over != (255.0, 255.0, 255.0, 0.0):
                # set the proper offsets on drawing the images
                if image is tophat:
                    new_y = y - py
                elif image is moustache:
                    new_y = h + y + py
                else:
                    new_y = y + py
                new_x = x + px

                # make sure the image is in the frame
                if 0 < new_x < frame.width and 0 < new_y < frame.height:
                    cv.Set2D(frame, new_y, new_x, over)


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
    if face:
        detect_feature(frame, face_cade, min_size=(60, 60))
    if eye:
        detect_feature(frame, eye_cade)
    if nose:
        detect_feature(frame, nose_cade)


def loop():
    """main video loop to render frames

    This loops until escape key is pressed, calling detect_features on each
    frame to look for faces.
    """
    global face, eye, nose

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
            face = False if face else True
        if key == 50:
            eye = False if eye else True
        if key == 51:
            nose = False if nose else True

        # get the frame, detect features, and show frame
        frame = cv.QueryFrame(source)
        detect_features(frame)
        cv.ShowImage(win_name, frame)

    cv.DestroyAllWindows()


if __name__ == '__main__':
    loop()
