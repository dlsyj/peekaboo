"""
The starts of a face detection and augmentation script using OpenCV
"""

import cv

# window name
name = "tests"
# video source (iSight is indexed at 0)
source = cv.CaptureFromCAM(0)

# create the GUI window
cv.NamedWindow(name)


def main():
    """main video loop to render frames

    This loops until escape key is pressed, pushing the camera images to the
    GUI window. On breaking from the loop, the windows are destroyed.
    """
    while True:
        # escape key to quit
        if cv.WaitKey(15) == 27:
            break

        # grab frame from camera and show in window
        frame = cv.QueryFrame(source)
        cv.ShowImage(name, frame)

    # close GUI Window
    cv.DestroyAllWindows()


if __name__ == '__main__':
    main()
