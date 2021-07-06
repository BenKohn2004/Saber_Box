# Virtual_Saber_Box

The Saber Box is a virtual directing aid. The program is contained within the Google Colab Notebook Virtual_Saber_Box. Virtual_Saber_Box_Compact is functionally identical with many of the Python scripts stored on Github.

The Saber Box is designed to determine the Right of Way of a saber action without blade contact. Specifically it is meant to differentiate between attack, counter-attack and preparation. These actions are the foundation of Saber, while also being some of the most subjective and difficult calls to make. The Saber Box is meant to offer a level of consistency in calling Right of Way actions and allow fencers to better plan and train with a repeatable and objective rule set.

Using the Saber Box.

The Saber Box can be run in one of three ways.
1.	Analyzing all files in a Sync folder on your Google Drive.
2.	Analyzing only the most recent file added to a Sync folder on your Google Drive
3.	A specific time stamp from a Youtube Clip.

To Run All the Files in the Sync Folder

1.	Create a folder in the top directory of your Google Drive titled ‘Sync’
2.	Place fencing clips in the folder to be analyzed
a.	Clips that are 1 to 4 seconds in length starting with the engarde position work best.
3.	Open Google Colab and find the Virtual_Saber_Box on my Github, BenKohn2004
4.	Ensure, run_entire_sync_folder == True, and the other two option are false.
5.	Runtime -> Run all
6.	Follow the prompt in the top cell to link your Google Drive
7.	The result can be found in the ‘Tracked Clips’ folder in the Sync directory.

To Run only the Most Recent File

1.	Create a folder in the top directory of your Google Drive titled ‘Sync’
2.	Open Google Colab and find the Virtual_Saber_Box on my Github, BenKohn2004
3.	Ensure, run_most_recent_clip == True, and the other two option are false.
4.	Runtime -> Run all
5.	Follow the prompt in the top cell to link your Google Drive
6.	After the program initially runs you should see ‘Waiting for a new file…’ at the bottom of the last cell.
7.	Place a file of a fencing clip into the ‘Sync’ folder for the program to recognize. The new file will run automatically.
8.	This is useful for recording a bout and viewing the results in realtime as shown at: https://youtu.be/qzMVuNEJ_6w
9.	The result can be found in the ‘Tracked Clips’ folder in the Sync directory.

To Run a Youtube Clip:
1.	Open Google Colab and find the Virtual_Saber_Box on my Github, BenKohn2004
2.	Ensure, use_youtube_link == True, and the other two option are false.
3.	Paste the youtube link with time stamp following ‘youtube_link =’ in the second cell.
4.	Youtube clips shorter than an hour tend to work better than longer youtube videos.
5.	Runtime -> Run all
6.	Follow the prompt in the top cell to link your Google Drive
7.	The result can be found in the ‘Tracked Clips’ folder in the Sync directory and the downloaded clip can be found in the ‘Sync’ folder.



A walkthrough on setting up the Virtual Saber Box can be found at: https://youtu.be/vGtuFIMhBZw



When using your own clip some tips to keep in mind are:
1.	Clips work best when centered and free from background clutter, specifically people
2.	Most of the clips I use are 1280x720 at 30 FPS.
3.	Video that has been compressed multiple times may be harder for the Computer Vision to track.
4.	Try to cut the video to 1 to 4 seconds and have the first 1/2 to 1 second be the fencers coming EnGarde.
5.	After running the model verify using the Tracked video that the colored dots track with the Bellguards. If the dots do not line up, then bad data was fed to the model determining Right of Way.
6.	There are various settings on the model. These are the Initial Parameters in the third cell.
7.	The model does not take into account blade contact so all actions will be viewed without regard to beats, parries or lines.
8.	Both fencer torsos must be visible at the start of the clip.
9.	Clips that fail to run will not stop the Google Colab notebook from running but also will not generate a Tracked video file.
  
Principles of Operation


Overall Operation

There are two main models used. One is a detection model based on YoloV4 and adapted through transfer learning to recognize bellguards, torsos and scoring boxes. The second model is a sequential Long Short Term Memory (LSTM) model to determine Right of Way using the output of the first model. 
There are two stages to the detection model. The first is the engarde_length phase where initial positions and values are established. This is meant to be a period of little motion so the Yolo model will work at its best. 

The second stage is focused on tracking. Tracking boxes are calculated for each tracked object and allowable boundaries are established for possible object positions. In practical terms there is a reasonable maximum speed for a bell guard in a fencing action. This is used as a maximum allowed motion of a tracking box from one frame to the next.

Much of the program is focused on determining the bell guard position. The exact mechanism used to determine the bell guard position depends on the confidence associated with each method. In general the hierarchy used to determine bell guard position is roughly:

1.	High Confidence detection within the tracking box
2.	Position based on detected motion between frames
3.	Expected position based on previous two frames.
4.	Linear Approximation based on confident positions

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
The torsos and scoring box use almost exclusively the detections from the Yolo model and tracking boxes. The bell guards use a hierarchy of possibilities based on the confidence of various inputs.

Motion Difference Tracking is used most often when the fencers are close to one another moving quickly. The quick motion is detrimental to both detection and human pose approximation. A small tracking box is used near the expected position of the bell guard. The difference between two frames is taken and sequential erosion and dilations are used to remove noise and highlight the contours of the bell guard. The most forward portion of the contour is then assumed to be the bell guard. It is done multiple times using smaller erosion values until a contour remains or the attempt fails.

If the previous attempts at determining a position fail, then the expected position is used based on the two previous positions and an assumed constant speed.


Scoring Box Lights

The score box detections can be unreliable depending on the type of box, background and lighting. The Saber Box attempts to detect the scoring box lights by detecting the score box and then looking for a change in color in the upper quadrants. 
Due to the variations in scoring machines, setups and lighting, this often fails and the Saber Box defaults to using the position of the bellguards to determine when the lights illuminate. When the bellguards are close to each other, the fencers are assumed to hit. This results in the Saber Box assuming that both fencers hit at the same time.

Creating the Clip Vector and Analyzing the Position

After the bell guard positions and the lights are determined a vector is created that represents the action. Each set within the vector will have the [left fencer horizontal position, the right fencer horizontal position, the left fencer’s light, the right fencer’s light]. This set will represent one frame in the fencing clip, therefore a 60 frame clip will have 240 values in sets of 4 representing each frame. The entire vector representing the fencing action is referred to as the clip_vector.
The difference between each clip vector position is found giving a speed for a given frame. The dimensions of the speed are in pixels per frame. The difference between each speed value is found resulting in an acceleration vector measured in pixels per frame squared. It is the acceleration clip vector that is used in the Right of Way model.
The clip vector is normalized to 30 frames per second (fps). If the clip vector created is more than 30 fps then it is downsampled to an approximate vector with 30 fps.


Right of Way Model

The Right of Way Model is a sequential LSTM meaning implying that order and position of each value in the vector is important. Intuitively, I suspect an attack can be viewed as a burst of acceleration and therefore when looked at on a timeline the fencer who first has a strong rise in acceleration will have the right of way.
The data is preprocessed by normalizing the values between -1 and 1. A maximum value is used based on the width of the frame. The clip then removes the values associated with the Engarde Positioning and a small amount of extra clips to ensure that the positioning is steady before the analysis starts. This is normally results in the first 15 frames being removed from the acceleration clip vector.

The clip is the analyzed using a previously generated model and a result is shown. The result is normalized so that the values for Left, Right or Simul add up to 1.

