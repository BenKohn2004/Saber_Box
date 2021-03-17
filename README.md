# Virtual_Saber_Box

Saber Box is a virtual directing aid. The program is contained within the Google Colab Notebook Virtual_Saber_Box. It will require you to connect to a Google Drive account.

A walkthrough on setting up the Virtual Saber Box can be found at: https://youtu.be/qzMVuNEJ_6w

Setting up the Virtual Saber Box requires four things.

1.	A Google Drive account with a Folder named ‘Sync’ in the top directory.
2.	The Virtual Saber Box notebook open in Google Colab.
3.	The Google Drive folder ‘Sync’, synced with a folder on your computer using Google Backup and Sync.
4.	A webcam that has the ‘Sync’ folder linked as the output location for captured video files.

To Use the Model:
1.	Verify that the Runtime on Google Colab is GPU and then run the Google Colab notebook Virtual Saber Box. Allow access to your Google Drive that has the ‘Sync’ Folder.
2.	Once the Google Colab Notebook has run until ‘Waiting for a new file...’, then add a video file to the ‘Sync’ folder. This can be done by copying and pasting a video file or generating a new one through a webcam.
3.	Google Colab should detect the new file and automatically generate a tracked file in the ‘Tracked Files’ folder in the Sync Folder.

When using your own clip some tips to keep in mind are:
1.	Clips work best when centered and free from background clutter, specifically people
2.	Most of the clips I use are 1280x720 at 30 FPS. Larger sizes work but may require more processing time.
3.	Video that has been compressed multiple times may be harder for the Computer Vision to track.
4.	Try to cut the video to 1 to 3 seconds and have the first 1/2 to 1 second be the fencers coming EnGarde.
5.	After running the model verify using the Tracked video that the colored dots track with the Bellguards. If the dots do not line up then bad data was fed to the model determining Right of Way.
6.	There are various settings on the model. As a default the fencers are assumed to hit once the Bellguards are close enough to each regardless of actual lights. This is due to difficulty tracking the scoring box with some setups.
7.	The model does not take into account blade contact so all actions will be viewed without regard to beats or parries.
8.	Both fencer torsos must be visible at the start of the clip.
9.	Clips that fail to run will not stop the Google Colab notebook from running but will not generate a Tracked video file.
  
  
  
Principles of Operation


Overall Operation

There are two main models used. One is a detection model based on Mask RCNN and adapted through transfer learning to recognize bellguards, torsos and scoring boxes. The second model is a sequential Long Short Term Memory (LSTM) model to determine Right of Way using the output of the first model. 
There are two stages to the detection model. The first is the engarde_length phase where initial positions and values are established. This is meant to be a period of little motion so the Mask RCNN model will work at its best. 

The second stage is focused on tracking. Tracking boxes are calculated for each tracked object and allowable boundaries are established for possible object positions. In practical terms there is a reasonable maximum speed for a bell guard in a fencing action.

Much of the program is focused on determining the bell guard position. The exact mechanism used to determine the bell guard position depends on the confidence associated with each method. In general the hierarchy used to determine bell guard position is roughly:

1.	High Confidence detection within the tracking box
2.	High Confidence based on Human Pose Approximation
3.	Position based on detected motion between frames
4.	Expected position based on previous two frames.
5.	Linear Approximation based on confident positions

Each frame is analyzed sequentially and a bell guard position is determined. The only exception is the linear approximation between points. The linear approximation can retroactively change the bell guard position if uncertain positions are surrounded by certain positions.


Order of the Operations

Duplicate Frames and Camera Motion

The main entry point into the program in through Process_Video_Clip.
The first major step is test_and_remove_duplicate_frames. This function is meant to remove duplicate frames and test for camera motion. This is done by finding the average difference between one frame to the next on a pixel by pixel basis. A low value implies the frame is effectively a duplicate most likely due to a video compression format. A high difference value between frames is interpreted as camera motion. The threshold for this is determined by using the average Hue Saturation Value (HSV) value over first the engarde_length.

Camera motion is detected to prevent the use of using motion based detection for frames while the camera is moving. Camera motion can also be detected and compensated for using a known stationary object such as the scorebox. The scorebox detections and tracking have been sufficiently unreliable that the scorebox position is not normally used to determine camera motion. A future intention is using Augmented Reality Tags (ARTags) while filming to create a known position and dimension.
After duplicate frames are removed frame with camera motion are noted, the frames are resaved for detection analysis.


Engarde Positioning

The engarde_length denotes the number of frames that will be used to establish initial positions and values. The engarde_length is normally about the first 10 frames. The detections for an object within a given area are averaged giving an approximate position from which it can start tracking after the Engarde Positioning. For example the fencer torsos may be required to be within the middle third of the frame on the left or right half respectively. The torso has been the most reliable detection so that is then used as a boundary requirement for bell guard detections. 
There is some robustness factored into the engarde detections. For example if the bell guard is not detected in the initial frames or outliers cause the average bell guard position to be outside an expected boundary then the bell guard position can be approximated using a given distance, based on the torso size, away from the torso. This robustness does not for the most part extend to torso detections since they are expected to be the most reliable and therefore if a torso is not detected within the engarde positioning the program will most likely fail.


Position Tracking

Tracking Boxes are created based on the confidence of the previously tracked positions. The tracking box expands to accommodate a larger region if the previous positions were uncertain and shrinks to its smallest size once a confident position is established.
The torsos and scoring box use almost exclusively the detections from the Mask RCNN model and tracking boxes. The bell guards use a hierarchy of possibilities based on the confidence of various inputs.

Human Pose Approximation is used when there is a high confidence in the wrist position of the human pose model. When the human pose is used the position is slightly forward and above the wrist position. The human pose is also used to determine if the bell guard has move behind the fencer’s knee. If the bell guard position is close to the knee then the knee position is used for the bell guard until the bell guard is again determined to be away from the knee. This is used to prevent a bell guard being obscured by the knee and the program assuming that the bell guard position continues to move backwards.

Motion Difference Tracking is used most often when the fencers are close to one another moving quickly. The quick motion is detrimental to both detection and human pose approximation. A small tracking box is used near the expected position of the bell guard. The difference between two frames is taken and sequential erosion and dilations are used to remove noise and highlight the contours of the bell guard. The most forward portion of the contour is then assumed to be the bell guard. It is done multiple times using smaller erosion values until a contour remains or the attempt fails.

If the previous attempts at determining a position fail, then the expected position is used based on the two previous positions and an assumed constant speed.
Scoring Box Lights

The score box detections can be unreliable depending on the type of box, background and lighting. Ideally the score box can be detected a change in the color or HSV of the lights from a default can detected. The default is calculated during the Engarde Positioning. Since the score box detections are unreliable it is difficult to know where to analyze for the light changes. As a work around one of the settings allows for an assumed two lights on when the bell guards are within are certain distance of each other based on the total width of the frame.
One of the planned future changes is to use an ARTag near the score box so that the tag can imply the location and size of the score box.


Creating the Clip Vector and Analyzing the Position

After the bell guard positions and the lights are determined a vector is created that represents the action. Each set within the vector will have the [left fencer horizontal position, the right fencer horizontal position, the left fencer’s light, the right fencer’s light]. This set will represent one frame in the fencing clip, therefore a 60 frame clip will have 240 values in sets of 4 representing each frame. The entire vector representing the fencing action is referred to as the clip_vector.
The difference between each clip vector position is found giving a speed for a given frame. The dimensions of the speed are in pixels per frame. The difference between each speed value is found resulting in an acceleration vector measured in pixels per frame squared. It is the acceleration clip vector that is used in the Right of Way model.
The clip vector is normalized to 30 frames per second (fps). If the clip vector created is more than 30 fps then it is downsampled to an approximate vector with 30 fps.


Right of Way Model

The Right of Way Model is a sequential LSTM meaning implying that order and position of each value in the vector is important. Intuitively, I suspect an attack can be viewed as a burst of acceleration and therefore when looked at on a timeline the fencer who first has a strong rise in acceleration will have the right of way.
The data is preprocessed by normalizing the values between -1 and 1. A maximum value is used based on the width of the frame. The clip then removes the values associated with the Engarde Positioning and a small amount of extra clips to ensure that the positioning is steady before the analysis starts. This is normally results in the first 15 frames being removed from the acceleration clip vector.

The clip is the analyzed using a previously generated model and a result is shown. The result is normalized so that the values for Left, Right or Simul add up to 1.

