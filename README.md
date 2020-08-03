# Saber_Box

Saber Box is a virtual directing aid. The program is contained within the Google Colab Notebook Saber_Box. It will require you to connect to a Google Drive account. 

You must put three files in the top directory of your Google Drive:

  1. The Clip you would like to Analyze
  2. The Model used to detect fencing object, mask_rcnn_bell_guard_cfg_0005.h5 
      https://drive.google.com/file/d/19mDLsh8WuBbPeOedpiXvIPUZqfYKuQXY/view?usp=sharing
  3. The Model used for determining Right of way, ROW_model.h5
      https://drive.google.com/file/d/1VIy9JAKNDpFQ1F5wDUBC9-47-i2LZnaF/view?usp=sharing
      
Open the Jupityr Notebook, Saber_Box in Google Colab. Change: video_filename = 'name.mp4' to the name of your clip.

Go to dropdown menu Runtime > Change Runtime Type and ensure that a GPU or TPU is selected.

Go to dropdown menu Runtime > Run All

Following the Prompt authorize access to your Google Drive and Paste the code into the Box. This will give the notebook access to the required files and creates folders for the fencing clip data.

The touch determination is displayed at the bottom of the Notebook.
