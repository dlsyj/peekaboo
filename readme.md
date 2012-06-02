Term Project Proposal: Face Tracking and Augmentation
=====================================================

Description
-----------

I will use OpenCV and its included Haar Cascade training data to detect facial
features from a camera feed. I'll then use the detected feature areas to map
images and shapes over the features to create some augmentation. For example,
once I am able to detect eyes, I could map images of cartoon eyes or circular
shapes the detected areas.

The end product should map detect a person's eyes, nose, and face, with the
eyes being detected as separate right and left areas instead of a pair of
areas. This should allow for versatility in overlaying the images, and with
the initial milestones involving drawing boxes, I'll get a good sense of how
accurate the training sets have made the program.

Milestones
----------

* ~~4/23 - Script set to pull video and analyze frames~~
* ~~4/30 - Face detection working, drawing a box around the head~~
* ~~5/7  - Eyes and mouth detection with boxes~~
* **5/9  - Midterm presentation**
* ~~5/14 - Ears and nose detection with boxes~~
* ~~5/21 - Switch boxes with image overlays~~
* ~~5/28 - Add keyboard shortcuts for different features~~
* **6/4  - Project presentation**

Sample Image
------------

![peekaboo](https://github.com/dzhurley/peekaboo/raw/master/data/sample.jpg)
