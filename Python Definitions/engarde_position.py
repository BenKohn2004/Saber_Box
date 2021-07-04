def engarde_position(bbox, capture_width, capture_height, engarde_length, frame_count, save_image_list):
  #Finds the initial positions to start tracking
  #Format of bbox[frame][roi], ([y1,x1,y2,x2], percent certainty, type)

  # Initializes the Bell Guard Positions
  # Position format [x,y]
  # Size format [[Width],[Height]]
  Left_Position = []
  Right_Position = []
  Bell_Guard_Size = [[],[]]
  Scoring_Box_Position = []
  Scoring_Box_Size = [[],[]]
  Left_Torso_Position = []
  Left_Torso_Size = [[],[]]
  Right_Torso_Position = []
  Right_Torso_Size = [[],[]]
  Left_Foot_Position = []
  Right_Foot_Position = []
  Foot_Size = [[],[]]
  All_Bell_Guard_Positions = []

  # sum_of_boxes is used to average the Left (x,y)(0), Right (x,y)(1), ScoreBox (x,y)(2), Left_Torso (x,y)(3), Right_Torso(x,y)(4),
  # Left_Foot(x,y)(5), Right_Foot(x,y),(6) values
  sum_of_boxes = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]

  # j represents the rois(specific bounding box) within the frame sorted by confidence score
  for j in range(len(bbox)):
    # The percent confidence for each region of interest (roi) is [i][j][1]
    # This uses the minimum value of the bbox (top-left) to determine Left, Right, Scorebox
    # The Bellguards must be centered within the frame, classified as Bellguards with a minimum confidence and have the correct color saturation
    # Adds values to the Left engarde box
    if (bbox[j][1] > bellguard_confidence and bbox[j][0][1] < int(capture_width*2/5) and bbox[j][0][0] < int(capture_height*3/4) and bbox[j][0][0] > int(capture_height*1/4) and bbox[j][2] == 1):
      test_result = saturation_test(bbox[j], frame_count, save_image_list)
      if verbose == True:
        display(f'The result of the saturation test for the Left Engarde Position is {test_result} at frame {frame_count}.')
      if test_result == True:
        #Appends x value:
        sum_of_boxes[0][0].append([bbox[j][0][1], bbox[j][1]])
        #Appends y value:
        sum_of_boxes[0][1].append([bbox[j][0][0], bbox[j][1]])
        #Appends x width value:
        Bell_Guard_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Bell_Guard_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    #Adds values to the Right engarde box
    elif (bbox[j][1] > bellguard_confidence and bbox[j][0][1] > int(capture_width*3/5) and bbox[j][0][0] < int(capture_height*3/4) and bbox[j][0][0] > int(capture_height*1/4) and bbox[j][2] == 1):
      # test_result = color_tester(bbox[i][j], i)
      test_result = saturation_test(bbox[j], frame_count, save_image_list)
      if verbose == True:
        display(f'The result of the saturation test for the Right Engarde Position is {test_result} at frame {frame_count}.')
      if test_result == True:
        #Appends x value:
        sum_of_boxes[1][0].append([bbox[j][0][1], bbox[j][1]])
        #Appends y value:
        sum_of_boxes[1][1].append([bbox[j][0][0], bbox[j][1]])
        #Appends x width value:
        Bell_Guard_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Bell_Guard_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    #Adds values to the ScoreBox Position
    elif (bbox[j][1] > 0.50 and bbox[j][0][1] > int(capture_width/4) and bbox[j][0][1] < int(capture_width*(3/4)) and bbox[j][2] == 2):
      #Appends x value:
      sum_of_boxes[2][0].append([bbox[j][0][1], bbox[j][1]])
      #Appends y value:
      sum_of_boxes[2][1].append([bbox[j][0][0], bbox[j][1]])  
      #Appends x width value:
      Scoring_Box_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
      #Appends y width value:
      Scoring_Box_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    # Adds values to the Left Foot Position
    elif (bbox[j][1] > foot_confidence and bbox[j][0][1] < int(capture_width*2/5) and bbox[j][0][0] > int(capture_height*2/4) and bbox[j][2] == 4):
      sum_of_boxes[5][0].append([bbox[j][0][1], bbox[j][1]])
      sum_of_boxes[5][1].append([bbox[j][0][0], bbox[j][1]])  
      Foot_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
      Foot_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    # Adds values to the Right Foot Position
    elif (bbox[j][1] > foot_confidence and bbox[j][0][1] > int(capture_width*2/5) and bbox[j][0][0] > int(capture_height*2/4) and bbox[j][2] == 4):
      sum_of_boxes[6][0].append([bbox[j][0][1], bbox[j][1]])
      sum_of_boxes[6][1].append([bbox[j][0][0], bbox[j][1]])  
      Foot_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
      Foot_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
    else:
      pass
    
    if bbox[j][2] == 1:
      All_Bell_Guard_Positions.append([int((bbox[j][0][3] + bbox[j][0][1])/2), int((bbox[j][0][2] + bbox[j][0][0])/2)])

  if verbose == True:
    display(f'The Left Foot Sum of Boxes at frame {frame_count} is:')
    display(sum_of_boxes[5])

  try:
    # Tests for cause of Left Engarde Position Failure
    if len(sum_of_boxes[0][0]) == 0:
      engarde_failure_test(bbox[j], bellguard_confidence, int(capture_width*2/5), int(capture_height*2/3), 'Left')
    # Tests for cause of Right Engarde Position Failure
    if len(sum_of_boxes[1][0]) == 0:
      engarde_failure_test(bbox[j], bellguard_confidence, int(capture_width*3/5), int(capture_height*3/4), 'Right')
  except:
    if verbose == True:
      display(f'There was an error in the engarde failure test and it was skipped.')

  # Finds the center points
  x_average_left = weight_average_list(sum_of_boxes[0][0])
  y_average_left = weight_average_list(sum_of_boxes[0][1])
  x_average_right = weight_average_list(sum_of_boxes[1][0])
  y_average_right = weight_average_list(sum_of_boxes[1][1])
  x_average_scorebox = weight_average_list(sum_of_boxes[2][0])
  y_average_scorebox = weight_average_list(sum_of_boxes[2][1])
  x_average_left_foot = weight_average_list(sum_of_boxes[5][0])
  y_average_left_foot = weight_average_list(sum_of_boxes[5][1])
  x_average_right_foot = weight_average_list(sum_of_boxes[6][0])
  y_average_right_foot = weight_average_list(sum_of_boxes[6][1])

  # Prevents a failure to detect the bellguard from failing to detect the torso
  # If the bellguard is unusually high or low then it is set to the height of the opposing BellGuard
  if (y_average_left < capture_height/5) or (y_average_left > capture_height*4/5):
    if verbose == True:
      display(f'The y_average_left was too high or low and was set to y_average_right.')
    y_average_left = y_average_right
  if (y_average_right < capture_height/5) or (y_average_right > capture_height*4/5):
    if verbose == True:
      display(f'The y_average_right was too high or low and was set to y_average_left.')
    y_average_right = y_average_left

  if verbose == True:
    display(f'The average left position is ({x_average_left},{y_average_left}).')
    display(f'The average right position is ({x_average_right},{y_average_right}).')

  # Bell_Guard_Size_average [Width, Height]
  Bell_Guard_Size_average = []
  # Appends the average scoring box width
  Bell_Guard_Size_average.append(average_list(Bell_Guard_Size[0]))
  # Appends the average scoring box height
  Bell_Guard_Size_average.append(average_list(Bell_Guard_Size[1]))

  # Finds the Torso Position After the Bell_Guard Position because the Bell_Guard is used as a constraint
  # j represents the rois(specific bounding box) within the frame sorted by confidence score
  for j in range(len(bbox)):
    # Adds values to the Left_Torso Position, similar requirements to Left guard
    # Minimum Torso confidence, on the left half of the screen, bottom of the box is below the bellguard, but also above 3 three times the bellguard height and is labeled torso
    if (bbox[j][1] > min_torso_confidence and bbox[j][0][1] < int(capture_width/2) and bbox[j][0][2] > (y_average_left - Bell_Guard_Size_average[1] * 2) \
        and bbox[j][0][2] < (y_average_left + 3*Bell_Guard_Size_average[1]) and bbox[j][2] == 3):
      test_result = True
      if test_result == True:
        # Appends x value:
        sum_of_boxes[3][0].append(bbox[j][0][1])
        #Appends y value:
        sum_of_boxes[3][1].append(bbox[j][0][0])
        #Appends x width value:
        Left_Torso_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
        #Appends y width value:
        Left_Torso_Size[1].append(bbox[j][0][2] - bbox[j][0][0])
      else:
        if verbose == True:
          display(f'The saturation test failed at frame {frame_count}.')
        else:
          pass
    # Adds values to the Right_Torso Position, similar requirements to Right guard
    # display(f'y_average_right-Bell_Guard_Size[1] * 2 is {(y_average_right-Bell_Guard_Size[1]*2)}.')

    if Bell_Guard_Size[1] == []:
      if verbose == True:
        display(f'The Bell Guard Height was not defined so it is set to a default of zero at frame {frame_count}.')
      Bell_Guard_Size[1] = 0

    if verbose == True:
      display(f'bbox[j][1] is {bbox[j][1]}.')
      display(f'bbox[j][0][1] is {bbox[j][0][1]}.')
      display(f'bbox[j][0][2] is {bbox[j][0][2]}.')
      display(f'y_average_right is {y_average_right}')
      display(f'Bell_Guard_Size[1] is {Bell_Guard_Size[1]}')
      display(f'(y_average_right-2*Bell_Guard_Size[1]) is {(y_average_right-2*Bell_Guard_Size[1])}.')
      display(f'bbox[j][1] is {bbox[j][1]}.')

    if (bbox[j][1] > min_torso_confidence and bbox[j][0][1] > int(capture_width/2) and bbox[j][0][2] > (y_average_right-2*Bell_Guard_Size_average[1]) and bbox[j][0][2] < (y_average_right + 3*Bell_Guard_Size_average[1]) and bbox[j][2] == 3):
      
      #Appends x value:
      sum_of_boxes[4][0].append(bbox[j][0][1])
      #Appends y value:
      sum_of_boxes[4][1].append(bbox[j][0][0])
      #Appends x width value:
      Right_Torso_Size[0].append(bbox[j][0][3] - bbox[j][0][1])
      #Appends y width value:
      Right_Torso_Size[1].append(bbox[j][0][2] - bbox[j][0][0])

  # Checks for a Failure to Detect the Left Torso
  if len(sum_of_boxes[3][0]) == 0:
    torso_failure_test(bbox, capture_width, capture_height, y_average_left, Bell_Guard_Size_average, 'Left', frame_count, min_torso_confidence)
  if verbose == True:
    display(f'Prior to torso failure test for right torso the y_average_left is {y_average_left}.')

  # Checks for a Failure to Detect the Right Torso
  if len(sum_of_boxes[4][0]) == 0:
    torso_failure_test(bbox, capture_width, capture_height, y_average_right, Bell_Guard_Size_average, 'Right', frame_count, min_torso_confidence) 
  if verbose == True:
    display(f'Prior to torso failure test for left torso the y_average_right is {y_average_right}.')

  #Finds the top left corner then moves the average point to the center
  x_average_left_torso = average_list(sum_of_boxes[3][0]) + average_list(Left_Torso_Size[0])/2
  y_average_left_torso = average_list(sum_of_boxes[3][1]) + average_list(Left_Torso_Size[1])/2
  x_average_right_torso = average_list(sum_of_boxes[4][0]) + average_list(Right_Torso_Size[0])/2
  y_average_right_torso = average_list(sum_of_boxes[4][1]) + average_list(Right_Torso_Size[1])/2

  if verbose == True:
    display(f'The average left engarde position is:({x_average_left},{y_average_left})')
    display(f'The average right engarde position is:({x_average_right},{y_average_right})')
    display(f'The average left torso is:({int(x_average_left_torso)},{int(y_average_left_torso)})')
    display(f'The average right torso is:({int(x_average_right_torso)},{int(y_average_right_torso)})')

  # scoring_box_size_average [Width, Height]
  scoring_box_size_average = []
  # Appends the average scoring box width
  scoring_box_size_average.append(average_list(Scoring_Box_Size[0]))
  # Appends the average scoring box height
  scoring_box_size_average.append(average_list(Scoring_Box_Size[1]))

  # left_torso_size_average [Width, Height]
  left_torso_size_average = []
  # Appends the average scoring box width
  left_torso_size_average.append(average_list(Left_Torso_Size[0]))
  # Appends the average scoring box height
  left_torso_size_average.append(average_list(Left_Torso_Size[1]))

  # right_torso_size_average [Width, Height]
  right_torso_size_average = []
  # Appends the average scoring box width
  right_torso_size_average.append(average_list(Right_Torso_Size[0]))
  # Appends the average scoring box height
  right_torso_size_average.append(average_list(Right_Torso_Size[1]))

  #Creates Padding for the EnGarde Tracking Box
  engarde_box_padding = int(capture_width/15)
  torso_padding = int(capture_width/20)
  foot_padding = int(capture_width/15)

  x_min_engardeL = int(x_average_left - engarde_box_padding)
  x_max_engardeL = int(x_average_left + engarde_box_padding)
  y_min_engardeL = int(y_average_left - engarde_box_padding)
  y_max_engardeL = int(y_average_left + engarde_box_padding)

  x_min_engardeR = int(x_average_right - engarde_box_padding)
  x_max_engardeR = int(x_average_right + engarde_box_padding)
  y_min_engardeR = int(y_average_right - engarde_box_padding)
  y_max_engardeR = int(y_average_right + engarde_box_padding)

  x_min_engardeScore = int(x_average_scorebox - engarde_box_padding)
  x_max_engardeScore = int(x_average_scorebox + engarde_box_padding)
  y_min_engardeScore = int(y_average_scorebox - engarde_box_padding)
  y_max_engardeScore = int(y_average_scorebox + engarde_box_padding)

  x_min_torsoL = int(x_average_left_torso - torso_padding)
  x_max_torsoL = int(x_average_left_torso + torso_padding)
  y_min_torsoL = int(y_average_left_torso - torso_padding*3/2)
  y_max_torsoL = int(y_average_left_torso + torso_padding*3/2)

  x_min_torsoR = int(x_average_right_torso - torso_padding)
  x_max_torsoR = int(x_average_right_torso + torso_padding)
  y_min_torsoR = int(y_average_right_torso - torso_padding*3/2)
  y_max_torsoR = int(y_average_right_torso + torso_padding*3/2)

  x_min_footL = int(x_average_left_foot)
  x_max_footL = int(x_average_left_foot + foot_padding*2)
  y_min_footL = int(y_average_left_foot - foot_padding)
  y_max_footL = int(y_average_left_foot + foot_padding)

  if verbose == True:
    display(f'The Left Foot xmin,xmax,ymin,ymax are {x_min_footL},{x_max_footL},{y_min_footL},{y_max_footL} at frame {frame_count}.')

  # x Foot Padding is large because both feet are detected and differentiated in the subsequent steps
  x_min_footR = int(x_average_right_foot - foot_padding*2)
  x_max_footR = int(x_average_right_foot)
  y_min_footR = int(y_average_right_foot - foot_padding)
  y_max_footR = int(y_average_right_foot + foot_padding)

  #Iterates through the first engarde_length frames and checks if there are rois in the expected engarde position
  for j in range(len(bbox)):
    y_center = int((bbox[j][0][0] + bbox[j][0][2])/2)
    x_center = int((bbox[j][0][1] + bbox[j][0][3])/2)
    # Checks for rois in the Left Engarde Position
    if (x_center > x_min_engardeL and x_center < x_max_engardeL and y_center > y_min_engardeL and y_center < y_max_engardeL and bbox[j][2] == 1):
      # display(f'The roi is in the left en garde position')
      Left_Position.append([x_center, y_center])
    # Checks for rois in the Right Engarde Position
    if (x_center > x_min_engardeR and x_center < x_max_engardeR and y_center > y_min_engardeR and y_center < y_max_engardeR and bbox[j][2] == 1):
      # display(f'The roi is in the right en garde position')
      Right_Position.append([x_center, y_center])
    # Checks for rois in the Scoring Box Position
    # if (x_center > x_min_engardeScore and x_center < x_max_engardeScore and y_center > y_min_engardeScore and y_center < y_max_engardeScore and bbox[j][2] == 2):
    if (bbox[j][2] == 2):
      Scoring_Box_Position.append([x_center, y_center])
      if verbose == True:
        display(f'Scoring_Box_Position appended is {x_center},{y_center} at frame {frame_count}')
    # Checks for rois in the Left Torso Position
    if (x_center > x_min_torsoL and x_center < x_max_torsoL and y_center > y_min_torsoL and y_center < y_max_torsoL and bbox[j][0][2] > y_average_left and bbox[j][2] == 3):
      Left_Torso_Position.append([x_center, y_center])
    # Checks for rois in the Right Torso Position 
    if (x_center > x_min_torsoR and x_center < x_max_torsoR and y_center > y_min_torsoR and y_center < y_max_torsoR and bbox[j][0][2] > y_average_right and bbox[j][2] == 3):
      Right_Torso_Position.append([x_center, y_center])
    # Checks for rois in the Left Foot Position 
    if (x_center > x_min_footL and x_center < x_max_footL and y_center > y_min_footL and y_center < y_max_footL and bbox[j][0][2] > y_average_left and bbox[j][0][1] > x_average_left_torso and bbox[j][2] == 4):
      Left_Foot_Position.append([x_center, y_center])
    # Checks for rois in the Right Foot Position 
    if (x_center > x_min_footR and x_center < x_max_footR and y_center > y_min_footR and y_center < y_max_footR and bbox[j][0][2] > y_average_right and bbox[j][0][1] < x_average_right_torso and bbox[j][2] == 4):
      Right_Foot_Position.append([x_center, y_center])

    # Tracking Bounding Box has unused brackets for Left and Right Torso
    Tracking_Bounding_Boxes_Temp = [[],[],[],[],[],[],[]]

    Tracking_Bounding_Boxes_Temp[0].append(x_min_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(x_max_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(y_min_engardeL)
    Tracking_Bounding_Boxes_Temp[0].append(y_max_engardeL)

    Tracking_Bounding_Boxes_Temp[1].append(x_min_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(x_max_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(y_min_engardeR)
    Tracking_Bounding_Boxes_Temp[1].append(y_max_engardeR)

    Tracking_Bounding_Boxes_Temp[2].append(x_min_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(x_max_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(y_min_engardeScore)
    Tracking_Bounding_Boxes_Temp[2].append(y_max_engardeScore)

    Tracking_Bounding_Boxes_Temp[5].append(x_max_footL)
    Tracking_Bounding_Boxes_Temp[5].append(x_max_footL)
    Tracking_Bounding_Boxes_Temp[5].append(y_min_footL)
    Tracking_Bounding_Boxes_Temp[5].append(y_max_footL)

    Tracking_Bounding_Boxes_Temp[6].append(x_max_footR)
    Tracking_Bounding_Boxes_Temp[6].append(x_max_footR)
    Tracking_Bounding_Boxes_Temp[6].append(y_min_footR)
    Tracking_Bounding_Boxes_Temp[6].append(y_max_footR)

    Tracking_Bounding_Boxes = Tracking_Bounding_Boxes_Temp

  # Creates a Tracking Bounding Boxes Variable if there are no Bounding Box detections
  if len(bbox) == 0:
    Tracking_Bounding_Boxes = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[],[],[0,0,0,0],[0,0,0,0]]

  # Tests for why a Torso Position is not Found
  if (len(Left_Torso_Position) == 0):
    torso_position_failure_test(bbox, engarde_length, x_min_torsoL, x_max_torsoL, y_min_torsoL, y_max_torsoL, y_average_left, 'Left', frame_count)    
  if (len(Right_Torso_Position) == 0):
    torso_position_failure_test(bbox, engarde_length, x_min_torsoR, x_max_torsoR, y_min_torsoR, y_max_torsoR, y_average_right, 'Right', frame_count)


  # Averages the Left and Right x,y positions for engarde
  # Left Bell Guard engarde position
  x = 0
  y = 0
  if len(Left_Position) > 0:
    for i in range(len(Left_Position)):
      x = x + Left_Position[i][0]
      y = y + Left_Position[i][1]
    x = int(x/(len(Left_Position)))
    y = int(y/(len(Left_Position)))
    Left_Position = [x,y]

  if verbose == True:
    display(f'Left_Position at Engarde is:')
    display(Left_Position)

  # Right Bell Guard engarde position
  x = 0
  y = 0
  if len(Right_Position) > 0:
    for i in range(len(Right_Position)):
      x = x + Right_Position[i][0]
      y = y + Right_Position[i][1]
    x = int(x/(len(Right_Position)))
    y = int(y/(len(Right_Position)))
    Right_Position = [x,y]

  if verbose == True:
    display(f'Right_Position at Engarde is:')
    display(Right_Position)

  # Scoring_Box engarde position
  x = 0
  y = 0
  if verbose == True:
    display(f'The lenth of score box positioning is {len(Scoring_Box_Position)}.')
  if len(Scoring_Box_Position) > 0:
    for i in range(len(Scoring_Box_Position)):
      x = x + Scoring_Box_Position[i][0]
      y = y + Scoring_Box_Position[i][1]
    x = int(x/(len(Scoring_Box_Position)))
    y = int(y/(len(Scoring_Box_Position)))
    Scoring_Box_Position = [x,y]

  if Scoring_Box_Position == [0,0]:
    Tracking_Bounding_Boxes_Temp[2] = [0,0,0,0]

  if verbose == True:
    display(f'Scoring_Box_Position at Engarde is:')
    display(Scoring_Box_Position)

  # Left_Torso engarde position
  x = 0
  y = 0
  if len(Left_Torso_Position) > 0:
    for i in range(len(Left_Torso_Position)):
      x = x + Left_Torso_Position[i][0]
      y = y + Left_Torso_Position[i][1]
    x = int(x/(len(Left_Torso_Position)))
    y = int(y/(len(Left_Torso_Position)))
    Left_Torso_Position = [x,y]

  if verbose == True:
    display(f'Left_Torso_Position at Engarde is:')
    display(Left_Torso_Position)

  # Right_Torso engarde position
  x = 0
  y = 0
  if len(Right_Torso_Position) > 0:
    for i in range(len(Right_Torso_Position)):
      x = x + Right_Torso_Position[i][0]
      y = y + Right_Torso_Position[i][1]
    x = int(x/(len(Right_Torso_Position)))
    y = int(y/(len(Right_Torso_Position)))
    Right_Torso_Position = [x,y]

  # Left_Foot engarde position
  x = 0
  y = 0
  if len(Left_Foot_Position) > 0:
    for i in range(len(Left_Foot_Position)):
      x = x + Left_Foot_Position[i][0]
      y = y + Left_Foot_Position[i][1]
    x = int(x/(len(Left_Foot_Position)))
    y = int(y/(len(Left_Foot_Position)))
    Left_Foot_Position = [x,y]

  # Right_Foot engarde position
  x = 0
  y = 0
  if len(Right_Foot_Position) > 0:
    for i in range(len(Right_Foot_Position)):
      x = x + Right_Foot_Position[i][0]
      y = y + Right_Foot_Position[i][1]
    x = int(x/(len(Right_Foot_Position)))
    y = int(y/(len(Right_Foot_Position)))
    Right_Foot_Position = [x,y]

  if verbose == True:
    display(f'Right_Torso_Position at Engarde is:')
    display(Right_Torso_Position)

  return (Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, Tracking_Bounding_Boxes, Left_Torso_Position, Right_Torso_Position, left_torso_size_average, right_torso_size_average, All_Bell_Guard_Positions, Left_Foot_Position, Right_Foot_Position, Foot_Size)