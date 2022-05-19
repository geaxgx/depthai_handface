# Face and hand tracking with DepthAI

Running Google Mediapipe Face Mesh and Hand Tracking models on [Luxonis DepthAI](https://docs.luxonis.com/projects/hardware/en/latest/) hardware (OAK-D, OAK-D lite, OAK-1,...). The hand tracking is optionnal and can be disabled by setting the argument `nb_hands` to 0.
<br>
**WIP**

<p align="center"><img src="media/yoga_eye_600_opt.gif" alt="Demo" /></p>

The models used in this repository are:
- [Mediapipe Blazeface](https://drive.google.com/file/d/1d4-xJP9PVzOvMBDgIjz6NhvpnlG9_i0S/preview), the short range version, for face detection. The distance face-camera must be < 2m.
- [Mediapipe Face Mesh](https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview) for face landmark detection(468 landmarks). I call this model the basic model in this document,
- [Mediapipe Face Mesh with attention](https://drive.google.com/file/d/1tV7EJb3XgMS7FwOErTgLU1ZocYyNmwlf/preview). This is an alternative to the previous model. In addition to the 468 landmarks, it can detect 10 more landmarks corresponding to the irises. Its predictions are more accurate around lips and eyes, at the expense of more compute (FPS on OAK-D ~10 frames/s). I call this model the attention model.
- The Mediapipe Palm Detection model (version 0.8.0) and Mediapipe Hand Landmarks models (version lite), already used in [depthai_hand_tracker](https://github.com/geaxgx/depthai_hand_tracker).

Note that, whenever possible, the post-processing of the models output has been integrated/concatenated to the models themselves, thanks to [PINTO's simple-onnx-processing-tools](https://github.com/PINTO0309/simple-onnx-processing-tools). Thus, Non Maximum Suppression for the face detection and palm detection models  as well as some calculation with the 468 or 478 face landmarks are done at the level of the models. The alternative would have been to do these calculations on the host or in a script node on the device (slower).


## Install

Install the python packages (depthai, opencv) with the following command:

```
python3 -m pip install -r requirements.txt
```

## Run

**Usage:**

```
->./demo.py -h
usage: demo.py [-h] [-i INPUT] [-a] [-2] [-n {0,1,2}] [-xyz] [-f INTERNAL_FPS]
               [--internal_frame_height INTERNAL_FRAME_HEIGHT] [-t [TRACE]]
               [-o OUTPUT]

optional arguments:
  -h, --help            show this help message and exit

Tracker arguments:
  -i INPUT, --input INPUT
                        Path to video or image file to use as input (if not
                        specified, use OAK color camera)
  -a, --with_attention  Use face landmark with attention model
  -2, --double_face     EXPERIMENTAL. Run a 2nd occurence of the face landmark Neural Network
                        to improve fps. Hand tracking is disabled.
  -n {0,1,2}, --nb_hands {0,1,2}
                        Number of hands tracked (default=2)
  -xyz, --xyz           Enable spatial location measure of palm centers
  -f INTERNAL_FPS, --internal_fps INTERNAL_FPS
                        Fps of internal color camera. Too high value lower NN
                        fps (default= depends on the model)
  --internal_frame_height INTERNAL_FRAME_HEIGHT
                        Internal color camera frame height in pixels
  -t [TRACE], --trace [TRACE]
                        Print some debug infos. The type of info depends on
                        the optional argument.

Renderer arguments:
  -o OUTPUT, --output OUTPUT
                        Path to output video file
```

**Some examples:**

- To run the basic face model with 2 hands max tracking:

    ```./demo.py``` 

- Same as above but with the attention face model:

    ```./demo.py -a``` 

- To run only the Face Mesh model (no hand tracking):

    ```./demo.py [-a] -n 0``` 

- If you want to track only one hand (instead of 2), you will get better FPS by running:

    ```./demo.py [-a] -n 1``` 

- Instead of the OAK* color camera, you can use another source (video or image) :

    ```./demo.py [-a] -i filename```

- To measure face and hand spatial location in camera coordinate system:

    ```./demo.py [-a] -xyz```

    The measure is made on the wrist keypoints and on a point of the forehead between the eyes.




|Keypress|Function|
|-|-|
|*Esc*|Exit|
|*space*|Pause|
|1|Show/hide the rotated bounding box around the hand|
|2|Show/hide the hand landmarks|
|3|Show/hide the rotated bounding box around the face|
|4|Show/hide the face landmarks|
|5|Show/hide hand spatial location (-xyz)|
|6|Show/hide the zone used to measure the spatial location (small purple square) (-xyz)|
|f|Switch between several face landmark rendering|
|f|Switch between several hand landmark rendering|
|b|Draw the landmarks on a black background|



## Credits
* [Google Mediapipe](https://github.com/google/mediapipe)
* Katsuya Hyodo a.k.a [Pinto](https://github.com/PINTO0309), the Wizard of Model Conversion !
* The video used in the illustration is [6 Eye Exercises: Tighten Droopy Eyelids and Reduce Wrinkles Around Eyes/ Blushwithme-Parmita](https://www.youtube.com/watch?v=X12oV-tVIpQ&list=PLrLHadod7vE821rcmUhM0LNrrn0DR9ZUb&index=4&ab_channel=Blushwithme-Parmita)  