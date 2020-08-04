# Saber_Box

Saber Box is a virtual directing aid. The program is contained within the Google Colab Notebook Saber_Box. It will require you to connect to a Google Drive account. 



You must put three files in the top directory of your Google Drive:

  1. The Clip you would like to Analyze
  2. The Model used to detect fencing object, mask_rcnn_bell_guard_cfg_0005.h5 
      https://drive.google.com/file/d/19mDLsh8WuBbPeOedpiXvIPUZqfYKuQXY/view?usp=sharing
  3. The Model used for determining Right of way, ROW_model.h5
      https://drive.google.com/file/d/1VIy9JAKNDpFQ1F5wDUBC9-47-i2LZnaF/view?usp=sharing
     
  The links above can be used to download the two models and then place them in the top directory of your Google Drive. A few fencing clips are also available to use as samples. 

To Use the Model:

1. Using this Github Repository, download the two models above

2. Open Google Drive and upload the two models. One model is about 250 MB so it may take a few minutes to upload.

3. Open Google Colab, Click the Github Tab, type in BenKohn2004 and the search icon.

4. Click on the Saber_Box.ipynb file.

5. Change: video_filename = '119.mp4' to the name of your clip.

6. Go to dropdown menu Runtime > Change Runtime Type and ensure that a GPU or TPU is selected.

7. Go to dropdown menu Runtime > Run All

8. Following the Prompt authorize access to your Google Drive and Paste the code into the Box. This will give the notebook access to the required files and creates folders for the fencing clip data.

9. The model will run and you can scroll down to see the Right of Way determination at the bottom.


When using your own clip some tips to keep in mind are:
  1. Clips work best when centered and free from background clutter, specifically people
  2. Most of the clips I use are 1280x720 at 30 FPS. Larger sizes work but may require more processing time.
  3. Video that has been compressed multiple times may be harder for the Computer Vision to track.
  4. Try to cut the video to 1 to 3 seconds and have the first 1/2 to 1 second be the fencers coming EnGarde.
  5. After running the model verify using the Overlay_Out video that the colored dots track with the Bellguards. If the dots do not line up then bad data was fed to the model determining Right of Way.
  6. There are various settings on the model. As a default the fencers are assumed to hit once the Bellguards are close enough to each regardless of actual lights. This is due to difficulty tracking the scoring box with some setups.
  7. The model does not take into account blade contact so all actions will be viewed without regard to beats or parries.
