def Bell_Guard_Position_Finding(bbox, capture_width, capture_height, positions, frame_count, left_torso_size_average, right_torso_size_average, engarde_length, previous_certainty, camera_steady, camera_motion_threshold, Exclusion_Areas, orig_img_worpt_starting_list):
  # Format positions = [Left_Position, Right_Position, Score_Box_Position, Left_Torso_Position, Right_Torso_Position]

  x_min = []
  x_max = []
  y_min = []
  y_max = []

  Left_Position = positions[0]
  Right_Position = positions[1]
  Scoring_Box_Position = positions[2]
  Left_Torso_Position = positions[3]
  Right_Torso_Position = positions[4]
  Left_Foot_Position = positions[5]
  Right_Foot_Position = positions[6]

  # Any of the First engarde_length position can be used since the engarde position is an averaged constant
  # Certainty is used here as a counter for how many times a bounding box does not fall in the tracking box
  # And increases the size of the bounding box based on each miss

  certainty = [0,0,0,0,0,0,0]
  if verbose == True:
    display(f'Previous Certainty at frame {frame_count - 1} is {previous_certainty}.')

  #Establishes Previous Positions to determine speed and expected positions
  previous_position_Left = Left_Position[-1]
  twice_previous_position_Left = Left_Position[-2]
  previous_position_Right = Right_Position[-1]
  twice_previous_position_Right = Right_Position[-2]
  previous_position_Scoring_Box = Scoring_Box_Position[-1]
  twice_previous_position_Scoring_Box = Scoring_Box_Position[-2]
  previous_position_Left_Torso = Left_Torso_Position[-1]
  twice_previous_position_Left_Torso = Left_Torso_Position[-2]
  previous_position_Right_Torso = Right_Torso_Position[-1]
  twice_previous_position_Right_Torso = Right_Torso_Position[-2]
  previous_position_Left_Foot = Left_Foot_Position[-1]
  twice_previous_position_Left_Foot = Left_Foot_Position[-2]
  previous_position_Right_Foot = Right_Foot_Position[-1]
  twice_previous_position_Right_Foot = Right_Foot_Position[-2]

  #Boxes are the bounding boxes for the current frame, passes less data to tracking function
  boxes = bbox

  # Tracking_Bounding_Boxes_Temp = [[],[],[],[],[]]
  Tracking_Bounding_Boxes_Temp = [[],[],[],[],[],[],[]]

  # Torso Positions are calculated prior to the BellGuard because they are an input to the bellguard position

  # Bellguard Position Tracking focuses on Tracking as opposed to detection
  
  # Left_Torso Position
  [current_position, certainty[3], Tracking_Bounding_Boxes_Left_Torso] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left_Torso, \
    twice_previous_position_Left_Torso, previous_certainty[3], 'Left_Torso', \
    frame_count, 'None', left_torso_size_average, capture_width, capture_height, \
    engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[3] = Tracking_Bounding_Boxes_Left_Torso
  Left_Torso_Position = current_position

  # Right_Torso Position
  [current_position, certainty[4], Tracking_Bounding_Boxes_Right_Torso] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Right_Torso, \
    twice_previous_position_Right_Torso, previous_certainty[4], "Right_Torso", \
    frame_count, 'None', right_torso_size_average, capture_width, capture_height, \
    engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[4] = Tracking_Bounding_Boxes_Right_Torso
  Right_Torso_Position = current_position

  # Left Position
  [current_position, certainty[0], Tracking_Bounding_Boxes_Left] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left, \
    twice_previous_position_Left, previous_certainty[0], 'Left_BellGuard', \
    frame_count, Left_Torso_Position, left_torso_size_average, capture_width, \
    capture_height, engarde_length, camera_steady, camera_motion_threshold, \
    Exclusion_Areas, orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[0] = Tracking_Bounding_Boxes_Left
  Left_Position = current_position

  #  Right Position
  [current_position, certainty[1], Tracking_Bounding_Boxes_Right] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Right, \
    twice_previous_position_Right, previous_certainty[1], 'Right_BellGuard', \
    frame_count, Right_Torso_Position, right_torso_size_average, capture_width, \
    capture_height, engarde_length, camera_steady, camera_motion_threshold, \
    Exclusion_Areas, orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[1] = Tracking_Bounding_Boxes_Right
  Right_Position = current_position

  # Scoring_Box Position
  [current_position, certainty[2], Tracking_Bounding_Boxes_Scoring_Box] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Scoring_Box, \
    twice_previous_position_Scoring_Box, previous_certainty[2], 'Scoring_Box', \
    frame_count, 'None', left_torso_size_average, capture_width, capture_height, \
    engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[2] = Tracking_Bounding_Boxes_Scoring_Box
  Scoring_Box_Position = current_position

  # Left Foot Position
  [current_position, certainty[5], Tracking_Bounding_Boxes_Left_Foot] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left_Foot, \
    twice_previous_position_Left_Foot, previous_certainty[5], 'Left_Foot', \
    frame_count, Left_Torso_Position, left_torso_size_average, capture_width, \
    capture_height, engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[5] = Tracking_Bounding_Boxes_Left_Foot
  Left_Foot_Position = current_position

  # Left Foot Position
  [current_position, certainty[5], Tracking_Bounding_Boxes_Left_Foot] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Left_Foot, \
    twice_previous_position_Left_Foot, previous_certainty[5], 'Left_Foot', \
    frame_count, Left_Torso_Position, left_torso_size_average, capture_width, \
    capture_height, engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[5] = Tracking_Bounding_Boxes_Left_Foot
  Left_Foot_Position = current_position

  # Right Foot Position
  [current_position, certainty[6], Tracking_Bounding_Boxes_Right_Foot] = \
    Bell_Guard_Position_Tracking(boxes, previous_position_Right_Foot, \
    twice_previous_position_Right_Foot, previous_certainty[6], 'Right_Foot', \
    frame_count, Right_Torso_Position, right_torso_size_average, capture_width, \
    capture_height, engarde_length, camera_steady, camera_motion_threshold, \
    'None', orig_img_worpt_starting_list)
  Tracking_Bounding_Boxes_Temp[6] = Tracking_Bounding_Boxes_Right_Foot
  Right_Foot_Position = current_position

  Tracking_Bounding_Boxes = Tracking_Bounding_Boxes_Temp

  if verbose == True:
    display(f'The Length of the Left and Right Positions after the Position Finding are: {len(Left_Position)} and {len(Right_Position)}.')
    display(f'At frame {frame_count} the certainty and previous certainty before linear approx analysis is:')
    display(f'{certainty} and {previous_certainty}')

  return (Left_Position, Right_Position, Scoring_Box_Position, Tracking_Bounding_Boxes, Left_Torso_Position, Right_Torso_Position, Left_Foot_Position, Right_Foot_Position, certainty)