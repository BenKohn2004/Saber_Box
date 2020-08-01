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

# Parameters:
min_torso_confidence = 0.80
bellguard_confidence = 0.65
# Provides for a higher confidence of bellguard detection
bellguard_confidence_high = 0.75
# Allows for a different required confidence for initial detection than tracking
bellguard_tracking_det_offset = 0.15
wrist_conf_min = 2
wrist_conf_high = 6
wrist_conf_very_high = 9
knee_conf_min = 3
# The Threshold for determining duplicate frames
duplicate_threshold_factor = 0.75
# The Threshold for determining camera motion
camera_motion_threshold_factor = 8
