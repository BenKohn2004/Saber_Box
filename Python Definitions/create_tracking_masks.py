def create_tracking_masks(previous_positions, certainty, frame_count, torso_size, width, height):
  #Creates Tracking Boxes that can be used to mask the image, ignoring parts that are not of interest
  #Format, Tracking_Boxes = [Left, Right, Scorebox], Left = [x_min, x_max, y_min, y_max]
  #Format, Previous Positions
          # previous_positions  = [[Left_Position[-1], Left_Position[-2]], \
          #                      [Right_Position[-1], Right_Position[-2]], \
          #                      [Scoring_Box_Position[-1], Scoring_Box_Position[-2]], \
          #                      [Left_Torso_Position[-1], Left_Torso_Position[-2]], \
          #                      [Right_Torso_Position[-1], Right_Torso_Position[-2]]]

          #Format, positions are [x,y]

  #Format, torso_position = [[Left_x,Lefty],[Right_x,Right_y]]
  #Format, torso_size = [[Lw,Lh], [Rw,Rh]]

  if verbose == True:
    display(f'Creating Tracking Masks...')
    display(f'The Previous Positions are:')
    display(previous_positions)
    display(f'The torso sizes are:')
    display(torso_size)

  frame_mask = []

  #Certainty is the number of times the bellguard has not been detected in previous frames
  #certainty_default is the minimum size of the tracking box
  certainty_default = int(width/16)
  #certainty_multiplier is how much the tracking box enlarges following a missed
  certainty_multiplier = int(width/80)
  y_limiter = 24

  #Max allowed speed of a bellguard in a single frame
  max_speed = int(width/48)

  if verbose == True:
    display(f'The length of the previous positions is: {len(previous_positions)}.')

  for i in range(len(previous_positions)):
    if verbose == True:
      display(f'The masking iteration for frame {frame_count} is {i}.')
    #FINDS THE LEFT MASKING BOX
    x_pos = previous_positions[i][0][0]
    y_pos = previous_positions[i][0][1]
    #Converts previous position into a speed
    x_speed = min(previous_positions[i][0][0] - previous_positions[i][1][0], max_speed)
    # Limits the maximum vertical speed with relation to x
    y_speed = min(previous_positions[i][0][1] - previous_positions[i][1][1], int(max_speed/y_limiter))

    if verbose == True:
      display(f'x and y position is ({x_pos},{y_pos}) and the speeds are ({x_speed},{y_speed}).')

    x_min = x_pos + (x_speed) - (certainty[i]*certainty_multiplier) - certainty_default
    x_max = x_pos + (x_speed) + (certainty[i]*certainty_multiplier) + certainty_default
    y_min = y_pos + (y_speed) - (certainty[i]*certainty_multiplier) - certainty_default
    y_max = y_pos + (y_speed) + (certainty[i]*certainty_multiplier) + certainty_default

    #Appends the mask to collection of tracked areas
    frame_mask.append([x_min, x_max, y_min, y_max])

  if verbose == True:
    display(f'The Frame Mask for frame {frame_count} is:')
    display(frame_mask)

  return (frame_mask)