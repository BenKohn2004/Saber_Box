# Establishes Initial Parameters
# For creating  a new .h5 detection model
train_model = False
# Provides additional comment feedback while running
verbose = True
# Crops each from to the tracking boxes
overlay_masking_boxes = False
# Adjusts the Reperesentative Dots based on camera motion implied by scoring box movement.
camera_motion_compensate = False
# Smooths the Bellguard positions
smooth_video_clip = False
# Assumes two lights if Bellguards are close to each other. Reduces dependency on Box detection.
assume_lights = True
# Ignores Lights from the Scorebox. Mitigates poor scorebox tracking.
ignore_box_lights = True
# Tests and Removes Duplicate Frames from the video
remove_duplicate_frames = True
# Downloads out and representative_out videos
download_videos = True
# Analyzes the Action
analyze_action = True
# Allows for Simple Usage
simplified = True
