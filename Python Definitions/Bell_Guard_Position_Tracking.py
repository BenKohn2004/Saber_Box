def Bell_Guard_Position_Tracking(boxes, previous_position, twice_previous_position, certainty, tracked_item, frame, Torso_Position, Torso_Size, capture_width, capture_height, engarde_length, camera_steady, camera_motion_threshold, Exclusion_Areas, orig_img_worpt_starting_list):
  # Tracks the position of items
  # tracked_item is needed since boxes only have the class of the item tracked, not the Left or Right
  # tracked_item Format: [0,1,2,3] = [Background, Bell_Guard, Score_Box, Torso]

  #Assumed inherent uncertainty
  certainty_default = int(capture_width/12)
  certainty_multiplier = int(capture_width/80)

  #Reduces the max value of y as compared to x
  y_limiter = 24

  # Max allowed speed of a bellguard in a single frame
  # Accounts for a position jump following the engarde positioning
  if frame < engarde_length + 3:
    max_speed = int(capture_width/64)
  else:
    max_speed = int(capture_width/24)

  #Sets the max speed to zero for Scoring Box if no camera motion
  if camera_steady[frame - 1] < camera_motion_threshold and tracked_item == 'Scoring_Box':
    max_speed = 0
    if verbose:
      display(f'Tracking the scorebox and camera is not moving. Max Speed set to zero at frame {frame - 1}.')

  # Converts previous position into a speed
  x_pos = int(previous_position[0])
  if verbose == True:
    display(f'previous_position is {previous_position} and twice_previous_position is {twice_previous_position}.')
  # x_speed = int(min(previous_position[0] - twice_previous_position[0], max_speed))

  x_speed = previous_position[0] - twice_previous_position[0]
  if x_speed > 0:
    x_speed = min(x_speed, max_speed)
  else:
    x_speed = max(x_speed, max_speed*-1)

  y_pos = int(previous_position[1])
  y_speed = int(min(previous_position[1] - twice_previous_position[1], int(max_speed/y_limiter)))
  y_speed = int(max(y_speed, int(max_speed*(-1)/y_limiter)))

  if (frame - 1)  == engarde_length and verbose == True:
    display(f'THe x_speed is {x_speed} and the y_speed is {y_speed} at the engarde length, frame {frame - 1}.')

  # Flips the tracking box to be between the two fencers
  if tracked_item == 'Left_BellGuard' or tracked_item == 'Left_Torso' or tracked_item == 'Left_Foot':
    horiz_flip = False
    if verbose == True:
      display(f'The horizontal flip is {horiz_flip} for the {tracked_item} at frame {frame - 1}.')
  elif tracked_item == 'Right_BellGuard' or tracked_item == 'Right_Torso' or tracked_item == 'Right_Foot':
    horiz_flip = True
    if verbose == True:
      display(f'The horizontal flip is {horiz_flip} for the {tracked_item} at frame {frame - 1}.')
  else:
    horiz_flip = False

  # Defines the tracking box
  expected_position = [(x_pos + x_speed),(y_pos + y_speed)]

  # Allows for more lenient box following engarde positioning
  if frame < (engarde_length + 3):
    padding = int(certainty*certainty_multiplier + certainty_default*1.5)
  else:
    padding = int(certainty*certainty_multiplier + certainty_default)
  # boundary_box_for_tracking = [int(padding*7/8), padding, padding, padding]
  boundary_box_for_tracking = [int(padding*16/8), padding, padding, padding]
  if verbose == True:
    display(f'The Boundary Box for the {tracked_item} is {boundary_box_for_tracking} using a certainty of {certainty} with an expected position of {expected_position} at frame {frame - 1}')
  tracking_box = create_boundary_box(expected_position, boundary_box_for_tracking, horiz_flip)
  positions = []

  boxes_temp = []
  #Filters out potential boxes based on Tracked Item, Confidence and Saturation of the Box
  if (tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard'):
    bell_certainty = certainty

    if verbose == True:
      display(f'The boxes being tested for exclusion are:')
      display(boxes)

    within_exclusion_box = False
    # Only Tests if Bell Guard confidence is not extra very high  
    for j in range(len(boxes)):
      if boxes[j][1] < bellguard_confidence_extra_very_high and use_Exclusion_Areas == True:
        # Tests if the box is in an exlusion area
        padding = capture_width/48
        exclusion_padding = [padding,padding,padding,padding]
        # Cycles through the Exclusion Areas
        for k in range(len(Exclusion_Areas)):
          if verbose == True:
            display(f'Exclusion_Areas[k] is {Exclusion_Areas[k]}.')
          Exculsion_Box = create_boundary_box(Exclusion_Areas[k],exclusion_padding, False)
          if verbose == True:
            display(f'boxes[j][0] is {boxes[j][0]}, boxes[j][1] is {boxes[j][1]}.')
          test_point = [int((boxes[j][0][3]+boxes[j][0][1])/2), int((boxes[j][0][2]+boxes[j][0][0])/2)]
          if verbose == True:
            display(f'The test_point is {test_point}.')
          within_tested_box = boundary_box_test(test_point, Exculsion_Box)
          if within_tested_box == True:
            within_exclusion_box = True
            if verbose == True:
              display(f'Within the exclusion box is {within_exclusion_box}.')
          else:
            if verbose == True:
              display(f'Within the exclusion box is {within_exclusion_box}.')

      # Allows Extra Very High Confidence regardless of exclusion areas
      if boxes[j][1] >= bellguard_confidence_extra_very_high:
         within_exclusion_box = False

      # Appends the Appropriate boxes to the boxes list to become potential positions 
      if (boxes[j][2] == 1) and (boxes[j][1] > (bellguard_confidence - bellguard_tracking_det_offset)) and within_exclusion_box == False:
        boxes_temp.append(boxes[j])

  elif (tracked_item == 'Left_Torso' or tracked_item == 'Right_Torso'):
    for j in range(len(boxes)):
      # Tests the torso for a drastic change in height from engarde positioning
      torso_height = boxes[j][0][2] - boxes[j][0][0]
      if verbose == True:
        display(f'The Torso Height at frame {frame - 1}, region of interest {j} is {torso_height} with initial height of {Torso_Size[1]}.')

      # Tests for minimum torso confidence and size compared to default
      if ((boxes[j][2] == 3) and (boxes[j][1] > min_torso_confidence) and torso_height > Torso_Size[1]*(2/3)):

        boundary_box = [int(Torso_Size[0]*4),int(Torso_Size[0]*4),int(Torso_Size[1]/4),int(Torso_Size[1]/4)]                                
        torso_box = create_boundary_box(expected_position, boundary_box, horiz_flip)                             

        # Allows for more lenient box following engarde positioning
        if frame < (engarde_length + 3):
          [x_min, x_max, y_min, y_max] = tracking_box
        else:
          [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, torso_box)
        x_center = int((boxes[j][0][1] + boxes[j][0][3])/2)
        y_center = int((boxes[j][0][0] + boxes[j][0][2])/2)
        torso_boundary_test = boundary_box_test([x_center,y_center], [x_min, x_max, y_min, y_max])
        if torso_boundary_test == True:
          boxes_temp.append(boxes[j])

  elif (tracked_item == 'Scoring_Box'):
    for j in range(len(boxes)):
      if (boxes[j][2] == 2):

       # Uses the Torso as a horizontal and vertical minimum
        # boundary_box_for_torso = [0, int(Torso_Size[0]*2), -1*int(Torso_Size[1]), int(Torso_Size[1]*2.0)]
        # scoring_box_box = create_boundary_box(ScoreBox_Position, boundary_box_for_torso, horiz_flip)
        [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, tracking_box)
        x_center = int((boxes[j][0][1] + boxes[j][0][3])/2)
        y_center = int((boxes[j][0][0] + boxes[j][0][2])/2)
        scorebox_boundary_test = boundary_box_test([x_center,y_center], [x_min, x_max, y_min, y_max])
        if scorebox_boundary_test == True:
          boxes_temp.append(boxes[j])

  elif (tracked_item == 'Left_Foot' or tracked_item == 'Right_Foot'):
    for j in range(len(boxes)):
      if ((boxes[j][2] == 4) and (boxes[j][1] > foot_confidence)):

        # Uses the Torso as a horizontal and vertical minimum
        boundary_box_for_torso = [0, int(Torso_Size[0]*2), -1*int(Torso_Size[1]), int(Torso_Size[1]*2.0)]
        torso_box = create_boundary_box(Torso_Position, boundary_box_for_torso, horiz_flip)
        [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, torso_box)
        x_center = int((boxes[j][0][1] + boxes[j][0][3])/2)
        y_center = int((boxes[j][0][0] + boxes[j][0][2])/2)
        foot_boundary_test = boundary_box_test([x_center,y_center], [x_min, x_max, y_min, y_max])
        if foot_boundary_test == True:
          boxes_temp.append(boxes[j])

  # Assigns boxes_temp to boxes
  boxes = boxes_temp

  # Creates points at the centers of the bounding boxes that are in this frame
  x_center = []
  y_center = []
  for i in range(len(boxes)):
    x_center.append(int((boxes[i][0][1] + boxes[i][0][3])/2))
    y_center.append(int((boxes[i][0][0] + boxes[i][0][2])/2))

  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    # Sets the values for the torso boundary box, limits Bellguard distance from Torso center
    boundary_box_for_torso = [int(Torso_Size[0]*0.20), int(Torso_Size[0]*3.25), int(Torso_Size[1]*.75), int(Torso_Size[1]*1.0)]
    # Uses the boundary box to create a box based on Left/Right and expected/previous position
    torso_box = create_boundary_box(Torso_Position, boundary_box_for_torso, horiz_flip)
    # Finds the overlap of multiple boxes to satisy multiple restrictions
    [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, torso_box)
    if verbose == True:
      display(f'The Torso_Size[0] is {Torso_Size[0]}, the Horizontal Flip is {horiz_flip} and Torso_Position is {Torso_Position}.')
  elif tracked_item == 'Left_Foot' or tracked_item == 'Right_Foot':
    # Uses the Torso as a horizontal and vertical minimum
    boundary_box_for_torso = [0, int(Torso_Size[0]*2), -1*int(Torso_Size[1]), int(Torso_Size[1]*2.0)]
    torso_box = create_boundary_box(Torso_Position, boundary_box_for_torso, horiz_flip)
    [x_min, x_max, y_min, y_max] = boundary_box_overlap(tracking_box, torso_box)
    if verbose == True:
      display(f'The Boundary Box for the {tracked_item} with the horiz_flip as {horiz_flip} at frame {frame -1} is:')
      display([x_min, x_max, y_min, y_max])
      display(f'The Torso Position is {Torso_Position}.')
      display(f'The Torso Box is {torso_box}.')
  else:
    [x_min, x_max, y_min, y_max] = tracking_box

  if verbose == True:
    display(f'The tracking box for the {tracked_item} at frame {frame - 1} is: {tracking_box}.')

  if (tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard'):
    if verbose == True:
      display(f'The torso box for the {tracked_item} at frame {frame - 1} is: {torso_box}.')
      display(f'The overlapping tracking box for the {tracked_item} at frame {frame - 1} is: {[x_min, x_max, y_min, y_max]}.')

  # Creates a list of positions within the bounding boxes
  for i in range(len(boxes)):
    center = [x_center[i], y_center[i]]
    tracking_result = boundary_box_test(center,tracking_box)
    # If the center point is within both boxes for Bellguards or tracking box for other items, then it is appended to positions
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      torso_result = boundary_box_test(center,torso_box)
      # if (frame - 1) == 49:
      #   display(f'For ({x_center[i]},{y_center[i]}), {boxes[i][1]}%, the tracking_result is {tracking_result} and the torso_result is {torso_result}.')
      # Allows for an incorrect engarde position for the bellguard
      if (frame - 1) > engarde_length + 3:
        if tracking_result == True and torso_result == True:
          positions.append([x_center[i],y_center[i], boxes[i][1]])
      else:
        # Only the torso results is required for the engarde positioning
        if torso_result == True:
          positions.append([x_center[i],y_center[i], boxes[i][1]])
    else:
      if tracking_result == True:
        positions.append([x_center[i],y_center[i], boxes[i][1]])

  # Maximum distance only applies if there are multiple bounding boxes within the tracking box
  maximum_distance_from_expected = int(capture_width/24)
  # Expected Position [x,y], Limits expected position in front of the fencer
  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    if verbose == True:
      display(f'The expected position is {expected_position} and Torso Position and size is {Torso_Position[0]} and {Torso_Size[0]}.')
    if (expected_position[0] > Torso_Position[0] + Torso_Size[0]*2.50) and tracked_item == 'Left_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was too far in front of the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0] + Torso_Size[0]*2.50), y_pos]
    if (expected_position[0] < Torso_Position[0]) and tracked_item == 'Left_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was behind the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0]), y_pos]
    if expected_position[0] < Torso_Position[0] - Torso_Size[0]*2.50 and tracked_item == 'Right_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was too far from the Torso, adjusting expected.')
        display(f'Torso_Position[0] is {Torso_Position[0]}, Torso_Size[0] is {Torso_Size[0]}, y_pos is {y_pos}.')
      expected_position = [int(Torso_Position[0] - Torso_Size[0]*2.50), y_pos]
    if (expected_position[0] > Torso_Position[0]) and tracked_item == 'Right_BellGuard':
      if verbose == True:
        display(f'At frame {frame - 1} the expected position of the {tracked_item} was behind the Torso, adjusting expected.')
      expected_position = [int(Torso_Position[0]), y_pos]

  #Assumed maximum distance from wrist to bellguard
  wrist_to_bellguard_max = int(Torso_Size[0]/8)

  #Sets Initial Conditions for Type of Tracking
  using_human_pose = False
  using_difference_images = False
  using_difference_images_normal_kernel = False
  using_expected = False
  using_position = False

  if verbose == True:
    display(f'The camera steady value for frame {frame - 1} is {camera_steady[frame - 1]}.')
    if camera_steady[frame - 1] >= camera_motion_threshold:
      if verbose == True:
        display(f'The camera is in motion and motion detection is less reliable.')

  # Determines the Bellguard Position based on number of detections, confidence, box location and motion
  if (len(positions)) == 0:
    if verbose == True:
      display(f'There where no positions found for the {tracked_item} at frame {frame - 1}.')
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      motion_difference_boundary = [int(Torso_Size[0]*5/8), int(Torso_Size[0] + 1), int(Torso_Size[0]/3), int(Torso_Size[0]/3)]
      if tracked_item == 'Left_BellGuard' and camera_steady[frame - 1] < camera_motion_threshold:
        boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
        position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, (capture_width/2560), 3, 4, orig_img_worpt_starting_list)
        box_test = boundary_box_test(position, boundary_box)
        if position != 'None' and box_test == True:
          using_difference_images_normal_kernel = True
        if position == 'None' or box_test == False:
          if verbose == True:
            display(f'Attempting to use a smaller kernel for motion difference tracking.')
          position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, (capture_width/640), 3, 4, orig_img_worpt_starting_list)
          box_test = boundary_box_test(position, boundary_box)
          # if position != 'None' and box_test == True:
          #   using_difference_images_normal_kernel = True
          if position == 'None' or box_test == False:
            if verbose == True:
              display(f'Attempting to use a smallest kernel for motion difference tracking.')
            position = motion_difference_tracking(frame, 'Left', boundary_box, capture_width, capture_height, (capture_width/320), 3, 4, orig_img_worpt_starting_list)
            if position == 'None':
              if verbose == True:
                display(f'Using the far Left Portion of the tracking Box')
              position = [expected_position[0] - motion_difference_boundary[0], expected_position[1]]
        # Adjusts the position if motion detection is too far out from the Torso
        if position[0] > Torso_Position[0] + Torso_Size[0]*2.50:
          if verbose == True:
            display(f'The motion detected position was too far from the torso and was adjusted')
          position[0] = int(Torso_Position[0] + Torso_Size[0]*2.50)
          # position = [Torso_Position[0] + Torso_Size[0]*2.25, position[1]]
        if verbose == True:
          display(f'The position for motion difference frame {frame - 1} is ({position})')
          display(f'The boundary box test limits are {motion_difference_boundary} for frame {frame - 1}.')
        # boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
        box_test = boundary_box_test(position, boundary_box)
        #Uses the Expected position if the motion difference is out of bounds
        if box_test == False:
          if verbose == True:
            display(f'Motion difference failed, using the Expected Position for the {tracked_item} for frame {frame - 1}.')
          position = expected_position
          using_expected = True
        else:
          if verbose == True:
            display(f'The motion difference position was used for the {tracked_item} at frame {frame - 1}.')
          using_difference_images = True
      elif tracked_item == 'Right_BellGuard' and camera_steady[frame - 1] < camera_motion_threshold:
        boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
        position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, (capture_width/2560), 3, 4, orig_img_worpt_starting_list)
        box_test = boundary_box_test(position, boundary_box)
        if position != 'None' and box_test == True:
          using_difference_images_normal_kernel = True
        if position == 'None' or box_test == False:
          if verbose == True:
            display(f'Attempting to use a smaller kernel for motion difference tracking.')
          position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, (capture_width/640), 3, 4, orig_img_worpt_starting_list)
          box_test = boundary_box_test(position, boundary_box)
          # if position != 'None' and box_test == True:
          #   using_difference_images_normal_kernel = True
          if position == 'None' or box_test == False:
            if verbose == True:
              display(f'Attempting to use the smallest kernel for motion difference tracking.')
            position = motion_difference_tracking(frame, 'Right', boundary_box, capture_width, capture_height, (capture_width/320), 3, 4, orig_img_worpt_starting_list)
            # Uses the Right Portion of the Motion Tracking Box if no motion tracking is found
            if position == 'None' or box_test == False:
              if verbose == True:
                display(f'Using the far Right Portion of the tracking Box, ({expected_position[0] + motion_difference_boundary[0]},{expected_position[1]})')
              position = [expected_position[0] + motion_difference_boundary[0], expected_position[1]]
        if position[0] < Torso_Position[0] - Torso_Size[0]*2.50:
          if verbose == True:
            display(f'The motion detected position ({position[0]},{position[1]}) was too far from the torso ({Torso_Position[0]},{Torso_Position[1]}), with a max of {Torso_Position[0] - Torso_Size[0]*2.50} and was adjusted')
          position[0] = int(Torso_Position[0] - Torso_Size[0]*2.50)
          # position = [Torso_Position[0] - Torso_Size[0]*2.25, position[1]]
        if verbose == True:
          display(f'The position for motion difference frame {frame - 1} is ({position})')
          display(f'The boundary box test limits are {motion_difference_boundary} for frame {frame - 1}.')
        boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
        box_test = boundary_box_test(position, boundary_box)
        # box_test = False
        if box_test == False:
          if verbose == True:
            display(f'Motion difference failed, using the Expected Position for the {tracked_item} for frame {frame - 1}.')
          position = expected_position
          using_expected = True
        else:
          if verbose == True:
            display(f'The motion difference position was used for the {tracked_item} at frame {frame - 1}.')
          using_difference_images = True
      else:
        if verbose == True:
          display(f'Too much camera motion, using expected position')
        position = expected_position
        using_expected = True    
    else:
      position = expected_position    

    # Criteria for Setting Certainty to zero preventing a linear appoximation adjustment of this point
    # Allows for motion difference to be certain if it uses the largest kernel
    if (using_difference_images_normal_kernel == True and position[1] < Torso_Position[1] + Torso_Size[1] and \
        abs(expected_position[0] - position[0]) < Torso_Size[0]/2 and camera_steady[frame - 1] < camera_motion_threshold):
      if using_difference_images == True:
        if verbose == True:
          display(f'Using difference images for frame {frame - 1} with no detected positions')
    # if (using_human_pose == True and fencer_data[0][2] > wrist_conf_high):
      certainty = 0
    else:
      certainty = certainty + 1

  # For a single detected Bellguard Position
  elif (len(positions)) == 1:
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      if verbose == True:
        display(f'There is one possible position, {positions[0]} for {tracked_item} in the tracking box for frame {frame - 1}.')
      # Allows for a large bounding box if bell guard confidence is very high
      if positions[0][2] > bellguard_confidence_extra_very_high:
        if verbose == True:
          display(f'Using Bell Guard Extra Very High confidence.')
        single_position_box = [int(Torso_Size[0]*9/8*(1+bell_certainty/4)), int(Torso_Size[0]*10/8*(1+bell_certainty/4)), int(Torso_Size[0]*12/8), int(Torso_Size[0]*12/8)]
      elif positions[0][2] > bellguard_confidence_very_high and positions[0][2] <= bellguard_confidence_extra_very_high:
        if verbose == True:
          display(f'Using Bell Guard Very High confidence.')
        single_position_box = [int(Torso_Size[0]*6/8*(1+bell_certainty/4)), int(Torso_Size[0]*6/8*(1+bell_certainty/4)), int(Torso_Size[0]*12/8), int(Torso_Size[0]*12/8)]
      else:
        single_position_box = [int(Torso_Size[0]*4/8*(1+bell_certainty/4)), int(Torso_Size[0]*5/8*(1+bell_certainty/4)), int(Torso_Size[0]*8/8), int(Torso_Size[0]*8/8)]
      if tracked_item == 'Left_BellGuard':
        boundary_box = create_boundary_box(expected_position, single_position_box, False)
      else:
        boundary_box = create_boundary_box(expected_position, single_position_box, True)
      box_test = boundary_box_test(positions[0], boundary_box)
      if verbose == True:
        display(f'The expected position for frame {frame - 1} is {expected_position}.')
        display(f'The single_position_box is {single_position_box} and the boundary box is {boundary_box}.')
      if box_test == True and positions[0][2] > bellguard_confidence_high:
        if verbose == True:
          display(f'The detected position was used for the {tracked_item} at frame {frame - 1}.')
        position = positions[0]
        using_position = True
      else:
        #Human Pose
        if verbose == True:
          display(f'Attempting to use Human Pose for the {tracked_item} at frame {frame - 1}')
        #Image Difference
        if verbose == True:
          display(f'Attempting to use Image Difference for the {tracked_item} at frame {frame - 1}')
        motion_difference_boundary = [int(Torso_Size[0]/8), int(Torso_Size[0]/2), int(Torso_Size[0]/4), int(Torso_Size[0]/4)]
        if tracked_item == 'Left_BellGuard':
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, False)
          diff_position = motion_difference_tracking(frame, 'Left', [x_min, x_max, y_min, y_max], capture_width, capture_height, 1, 1, 2, orig_img_worpt_starting_list)
          if diff_position == 'None':
            diff_position = motion_difference_tracking(frame, 'Left', [x_min, x_max, y_min, y_max], capture_width, capture_height, 2, 1, 2, orig_img_worpt_starting_list)
        else:
          #Right Bellguard is assumed
          boundary_box = create_boundary_box(expected_position, motion_difference_boundary, True)
          diff_position = motion_difference_tracking(frame, 'Right', [x_min, x_max, y_min, y_max], capture_width, capture_height, 1, 1, 2, orig_img_worpt_starting_list)
          if diff_position == 'None':
            diff_position = motion_difference_tracking(frame, 'Right', [x_min, x_max, y_min, y_max], capture_width, capture_height, 2, 1, 2, orig_img_worpt_starting_list)
        box_test = boundary_box_test(diff_position, motion_difference_boundary)
        if box_test == True and diff_position != 'None':
          position = diff_position
          using_difference_images = True
        else:
          #Expected Position
          position = expected_position
          using_expected = True
        if verbose == True:
          display(f'The position for motion difference frame {frame - 1} is ({position})')
          display(f'The motion_difference_boundary test limits are {motion_difference_boundary} for frame {frame - 1}.')

      # Designed to catch an engarde position that is outside the tracking box
      if frame < (engarde_length + 3) and position == twice_previous_position:
        position = positions[0]

      #Sets Certainty Box
      # if (using_human_pose == True and fencer_data_side[0][2] > wrist_conf_high) or (using_position == True):
      if using_position == True:
        certainty = 0
        if verbose == True:
          display(f'Certainty set to zero for frame {frame - 1} for the {tracked_item}.')
      else:
        certainty = certainty + 1

    else:
      position = positions[0]

  # Multiple bounding boxes within the tracking box
  elif (len(positions)) > 1:
    if verbose == True:
      display(f'Multiple Bounding Boxes Detected for the {tracked_item} at frame {frame - 1}')
    # One set of conditions is used for Bell_Guards and another for all else
    if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
      if positions[0][2] > bellguard_confidence_high:
        if verbose == True:
          display(f'Using Multiple Box Determination for the {tracked_item} at frame {frame - 1}.')
        human_pose_boundary = [int(Torso_Size[0]*3/4), int(Torso_Size[0]), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
        position = multiple_box_determination(expected_position, positions, [human_pose_boundary[0], human_pose_boundary[1]], bellguard_confidence, horiz_flip)
        using_position = True
      else:
        position = expected_position
        using_expected = True
        if verbose == True:
          display(f'The Human Pose Box Test failed for the {tracked_item} at frame {frame - 1}, using expected position.')
      within_distance_from_expected = []
      for i in range(len(positions)):
        expected_box = [int(Torso_Size[0]/2*(1+bell_certainty/4)), int(Torso_Size[0]*(1+bell_certainty/4)), int(Torso_Size[0]/2*(1+bell_certainty/4)), int(Torso_Size[0]/6)]
        if tracked_item == 'Left_BellGuard':
          boundary_box = create_boundary_box(expected_position, expected_box, False)
        else:
          boundary_box = create_boundary_box(expected_position, expected_box, True)
        box_test = boundary_box_test(positions[i], boundary_box)
        if box_test:
          within_distance_from_expected.append(positions[i])

        # Uses the most confident, i.e. the first position in the list
        if len(within_distance_from_expected) > 0:
          position_boundary = [int(Torso_Size[0]/4), int(Torso_Size[0]/2), int(Torso_Size[0]/2), int(Torso_Size[0]/2)]
          position = multiple_box_determination(expected_position, positions, [position_boundary[0], position_boundary[1]], bellguard_confidence, horiz_flip)
          certainty = 0
          using_position = True
        else:
          # If the length of within_distance_from_expected is zero
          if verbose == True:
            display(f'Error occured finding a position within the required distance and the {tracked_item} set to expected position at frame {frame - 1}.')
            display(f'The expected position is {expected_position}, while the expected box is {expected_box}.')
          position = [(x_pos + x_speed),(y_pos + y_speed)]
          using_expected = True

      #Sets Certainty Box
      # if (using_human_pose == True and fencer_data_side[0][2] > wrist_conf_high) or (using_position == True):
      if using_position == True:
        if verbose == True:
          display(f'Confidence for the {tracked_item} is High so the certainty is set to zero.')
        certainty = 0
      else:
        if verbose == True:
          display(f'Confidence for the {tracked_item} is Low so the certainty is incremented higher.')
        certainty = certainty + 1

    elif tracked_item == 'Left_Foot' or tracked_item == 'Right_Foot':
      # Max and Min are based on the first value of the set therefore in this case max and min refer to the x position
      if horiz_flip == False:
        position = max(positions)
      else:
        position = min(positions)
      certainty = 0

    # If the tracked item is not a bell_guard
    else:
      # Uses the most confident position within the tracking box
      position = positions[0]
      # Sets Certainty for Torso and Box back to Zero if detected.
      certainty = 0

  if tracked_item == 'Left_BellGuard' or tracked_item == 'Right_BellGuard':
    if verbose == True:
      display(f'The position of the {tracked_item} at frame {frame - 1} is {position}.')

  return (position, certainty, [x_min, x_max, y_min, y_max])