def process_video_clip(file_name, touch_folder, remove_duplicate_frames):
  # Processes the video

  display(f'The video file_name is: {file_name}')

  # Initiates Conditions
  score_box_empty = False
  right_torso_empty = False
  left_torso_empty = False
  left_position_empty = False
  right_position_empty = False
  left_foot_empty = False
  right_foot_empty = False

  save_image_list = []
  Left_Position = []
  Right_Position = []
  Scoring_Box_Position = []
  Left_Torso_Position = []
  Right_Torso_Position = []
  Left_Foot_Position = []
  Right_Foot_Position = []
  left_torso_size_average = []
  right_torso_size_average = []
  Tracking_Bounding_Boxes = []
  All_Bell_Guard_Positions = []
  Exclusion_Areas = []
  bbox_clip = []
  scorebox_bbox_list = []
  ScoreBox_Lights = []

  skip_frames = False
  skip_frame_counter = 0
  frames_to_skip = 300
  number_of_frames_skipped = 0

  clip_vector_previous = []

  # Removes and Recreates Save, Original and Original without Repeats folders to ensure empty and moves the clip to the appropriate directory
  [ROOT_DIR, VIDEO_DIR, VIDEO_SAVE_DIR, VIDEO_ORIG_DIR] = setup_clip_directories()

  engarde_length = 10

  video_clip_dir = '/content/drive/My Drive/Sync/'
  [original_image_list, fps] = read_video(file_name, video_clip_dir)

  # Removes Duplicates and Detects Camera motion in Frames
  [camera_steady, duplicate_threshold, camera_motion_threshold, original_image_list, original_image_worpt_list] \
  = test_and_remove_duplicate_frames(original_image_list, engarde_length, remove_duplicate_frames)

  width = original_image_worpt_list[0].shape[1]
  height = original_image_worpt_list[0].shape[0]

  # Determines the Starting Frame of the file
  [starting_frame, total_frames] = determine_starting_frame(original_image_worpt_list, width, height)

  start_time = starting_frame/fps
  end_time = total_frames/fps

  orig_img_worpt_starting_list = original_image_worpt_list[starting_frame:]

  if total_frames == 0:
    display(f'ERROR: The Video Clip selected has no frames.')
  display(f'The total number of frames in the video are: {total_frames}')

  try:
    if not os.path.exists(VIDEO_SAVE_DIR):
          os.makedirs(VIDEO_SAVE_DIR)
  except OSError:
    print ('Error: Creating directory of data')

  frame_count = 0

  if verbose == True:
    display(f'The capture width is: {width}')
    display(f'The capture height is: {height}')
  
  # Continues until it breaks by not finding a return from attempting to read a frame capture

  # Creates a variable that allows looping if engarde positioning is not effective
  perform_engarde_positioning = True
  engarde_offset = 0
  # The number of frames that is incremented when an engarde positioning fails
  offset_increment = 10

  close_bellguards = False
  too_few_frames = False
  frames_after_light_iterator = 0
  frames_after_light_max = 15
  minimum_frames_to_run = 20

  if (total_frames - starting_frame) > minimum_frames_to_run:
    display(f'The remaining frames are {(total_frames - starting_frame)}, which is greater than {minimum_frames_to_run}.')
  else:
    display(f'The remaining frames are {(total_frames - starting_frame)}, which is fewer than {minimum_frames_to_run}.')

  # if ((total_frames - starting_frame) =< minimum_frames_to_run):
  if ((total_frames - starting_frame - engarde_offset) <= minimum_frames_to_run):
    display(f'Too Few Frames to run {file_name}')
    too_few_frames = True


  # Runs for 5 frames after the lights turn on and required at least 30 frames total
  while (total_frames - starting_frame) > minimum_frames_to_run and (frame_count) < len(orig_img_worpt_starting_list):
    # Iterator
    frame_count += 1
    #Creates Bounding Box List
    frame = orig_img_worpt_starting_list[frame_count - 1]
    frames = [frame]

    # Runs the Detection Model for Bellguard, Torso, Scoring_Box
    [bbox, frame] = yolov4_run_image(frame, True)
    save_image_list.append(frame)

    if export_scorebox_image == True or use_scorebox_lights == True:
      # Gets the BBox of the scoreboxes
      scorebox_bbox_list_temp = scorebox_bbox(bbox)
      # Scorebox_bbox_list is a list of each scorebox detection bbox for each frame.
      scorebox_bbox_list.append(scorebox_bbox_list_temp)

    bbox_clip.append(bbox)

    if (frame_count <= (engarde_length + engarde_offset)):

      # Uses a frame in the middle of positioning to determine the time for frame analysis

      # Performs Engarde Positioning for a frame
      [Left_Position_Temp, Right_Position_Temp, Scoring_Box_Position_Temp, scoring_box_size_average, Tracking_Bounding_Boxes_Temp, \
        Left_Torso_Position_Temp, Right_Torso_Position_Temp, left_torso_size_average_Temp, right_torso_size_average_Temp, \
       All_Bell_Guard_Positions_Temp, Left_Foot_Position, Right_Foot_Position, Foot_Size] \
        = engarde_position(bbox, width, height, engarde_length + engarde_offset, frame_count-1, save_image_list)

      Left_Position.append(Left_Position_Temp)
      Right_Position.append(Right_Position_Temp)

      Scoring_Box_Position.append(Scoring_Box_Position_Temp)
      Left_Torso_Position.append(Left_Torso_Position_Temp)
      Right_Torso_Position.append(Right_Torso_Position_Temp)
      left_torso_size_average.append(left_torso_size_average_Temp)
      right_torso_size_average.append(right_torso_size_average_Temp)
      if verbose == True:
        display(f'At frame {frame_count} the tracking Bounding Boxes Temp is:')
        display(Tracking_Bounding_Boxes_Temp)
      Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)
      # Creates a List of each Bell Guard Position in each engarde positioning frame. Used for excluding false positives
      All_Bell_Guard_Positions.append(All_Bell_Guard_Positions_Temp)

    # At the Engarde_Legnth Frame.
    if (frame_count == engarde_length + engarde_offset):

      if verbose == True:
        display(f'Commencing the Engarde Length Processing.')
      # tracked_items = [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position, left_torso_size_average, right_torso_size_average]
      tracked_items = [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position, left_torso_size_average, right_torso_size_average, Left_Foot_Position, Right_Foot_Position]

      if verbose == True:
        display(f'The Scoring Box Position is:')
        display(Scoring_Box_Position)
      if max(Scoring_Box_Position) == []:
        if verbose == True:
          display(f'The Scoring Box Position was empty so a default was used.')
        score_box_empty = True
        tracked_items[2] = [[width/2, height/2], [width/2, height/2], [width/2, height/2]]

      # Tests for empty Positions
      if max(Right_Torso_Position) == []:
        right_torso_empty = True

      if max(Left_Torso_Position) == []:
        left_torso_empty = True

      if max(Left_Position) == [] and max(Left_Torso_Position) != []:
        left_position_empty = True

      if max(Right_Position) == [] and max(Right_Torso_Position) != []:
        right_position_empty = True
      
      # Replaces [0,0] with [] for torso width and height
      for k in range(len(tracked_items)):
        for j in range(len(tracked_items[k])):
          if tracked_items[k][j] == [0,0]:
            tracked_items[k][j] = []

      for k in range(len(tracked_items)):
        if verbose == True:
          display(f'k is {k}.')
          display(tracked_items[k])
        try:
          tracked_items[k] = [item for item in tracked_items[k] if item != []]
          tracked_items[k] = np.hstack(tracked_items[k])
          if verbose == True:
            display(f'The length of len(tracked_items[k]) is {int(len(tracked_items[k])/2)}.')
          tracked_items[k] = tracked_items[k].reshape(int(len(tracked_items[k])/2), 2)
          tracked_items[k] = np.median(tracked_items[k], axis = 0)
          tracked_items[k] = [int(tracked_items[k][0]), int(tracked_items[k][1])]
          if verbose == True:
            display(f'The tracked item position {tracked_items[k]}.')
        except:
          if verbose == True:
            display(f'Failure to detect the tracked item {k}, {tracked_items[k]} during the engarde positioning.')
          tracked_items[k] = [0,0]

      if right_torso_empty == True:
        # if verbose == True:
        display(f'The Right Torso Position was empty so a default based on the Right BellGuard was used.')
        display(f'tracked_items[1] is {tracked_items[1]}, left_torso_size_average is {left_torso_size_average}.')
        torso_position_default = [tracked_items[1][0] + int(left_torso_size_average[0][0]*3/4), tracked_items[1][1] - int(left_torso_size_average[0][1]/4)]
        tracked_items[4] = torso_position_default
        # Sets the Right Tracking Bounding Box
        # Sets the Right Torso Average Size to the Left Average Size
        right_torso_size_average = left_torso_size_average
        if verbose == True:
          display(f'As a comparison, the Score Box Position is:')
          display(Scoring_Box_Position)
          display(f'The Right Torso Position is:')
          display(Right_Torso_Position)
          display(f'As a comparison, the tracked_items[2] is:')
          display(tracked_items[2])
          display(f'The tracked_items[4] is:')
          display(tracked_items[4])
        # Sets Right Torso Empty to False after a Default it used
        right_torso_empty = False

      if left_torso_empty == True:
        if verbose == True:
          display(f'The Left Torso Position was empty so a default based on the Left BellGuard was used.')
          display(f'tracked_items[0] is {tracked_items[0]}, right_torso_size_average is {right_torso_size_average}.')
        torso_position_default = [tracked_items[0][0] - int(right_torso_size_average[1][0]*3/4), tracked_items[0][1] - int(right_torso_size_average[0][1]/4)]
        tracked_items[3] = torso_position_default
        # Sets the Left Torso Average Size to the Left Average Size
        left_torso_size_average = right_torso_size_average
        if verbose == True:
          display(f'As a comparison, the Score Box Position is:')
          display(Scoring_Box_Position)
          display(f'The Left Torso Position is:')
          display(Left_Torso_Position)
          display(f'As a comparison, the tracked_items[1] is:')
          display(tracked_items[1])
          display(f'The tracked_items[3] is:')
          display(tracked_items[3])
        left_torso_empty = False

      if left_position_empty == True:
        if verbose == True:
          display(f'The Left  Position was empty so a default based on the Left Torso was used.')
        left_torso_size_average_temp = average_list_without_null(left_torso_size_average)
        left_position_default = [int(tracked_items[3][0] + left_torso_size_average_temp[0]), int(tracked_items[3][1] + left_torso_size_average_temp[0]/4)]
        # left_position_default = [tracked_items[3][0] + int(left_torso_size_average[0][0]), tracked_items[3][1] + int(left_torso_size_average[0][0]/4)]
        tracked_items[0] = left_position_default

      if right_position_empty == True:
        if verbose == True:
          display(f'The Right  Position was empty so a default based on the Right Torso was used.')
          display(f'The tracked_items[4] is {tracked_items[4]}.')
          display(f'The right_torso_size_average is {right_torso_size_average}.')
        right_torso_size_average_temp = average_list_without_null(right_torso_size_average)
        right_position_default = [int(tracked_items[4][0] - right_torso_size_average_temp[0]), int(tracked_items[4][1] + right_torso_size_average_temp[0]/4)]
        tracked_items[1] = right_position_default

      [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position, Left_Foot_Position, Right_Foot_Position] = [[],[],[],[],[],[],[]]

      if verbose == True:
        display(f'The length of the tracked items is {len(tracked_items)}')

      # Builds the Positions of the Tracked Items, iterator is solely to append multiple times
      for i in range(engarde_length + engarde_offset):
        Left_Position.append(tracked_items[0])
        Right_Position.append(tracked_items[1])
        Scoring_Box_Position.append(tracked_items[2])
        Left_Torso_Position.append(tracked_items[3])
        Right_Torso_Position.append(tracked_items[4])
        Left_Foot_Position.append(tracked_items[7])
        Right_Foot_Position.append(tracked_items[8])

      if verbose == True:
        display(f'The Left Foot Position is:')
        display(Left_Foot_Position)

      padding = width/24
      exclusion_box = [padding,padding,padding,padding]
      Left_Exculsion_Box = create_boundary_box(Left_Position[0],exclusion_box, False)
      Right_Exculsion_Box = create_boundary_box(Right_Position[0],exclusion_box, False)

      for m in range(len(All_Bell_Guard_Positions)):
        for n in range(len(All_Bell_Guard_Positions[m])):
          within_Left_Region = boundary_box_test(All_Bell_Guard_Positions[m][n], Left_Exculsion_Box)
          within_Right_Region = boundary_box_test(All_Bell_Guard_Positions[m][n], Right_Exculsion_Box)
          if within_Left_Region == False and within_Right_Region == False:
            Exclusion_Areas.append(All_Bell_Guard_Positions[m][n])

      # Removes duplicate Exclusion Areas that are within a given distance of each other
      Exclusion_Areas = exclusion_area_simplification_recursion(Exclusion_Areas, width/80)

      if verbose == True:
        display(f'The Exclusion areas are:')
        display(Exclusion_Areas)

      left_torso_size_average = tracked_items[5]
      if right_torso_empty == False:
        right_torso_size_average = tracked_items[6]
      elif right_torso_empty == True:
        right_torso_size_average = tracked_items[5]
      torso_size = [left_torso_size_average, right_torso_size_average]

      if verbose == True:
        display(f'The left_torso_size_average is {left_torso_size_average} and the right_torso_size_average is {right_torso_size_average}.')
        display(f'The Right_Torso_Position[0] Prior to testing is: {Right_Torso_Position[0]}. ')

      if (Left_Position[0] == [0,0] or Right_Position[0] == [0,0] or Left_Torso_Position[0] == [0,0] or Right_Torso_Position[0] == [0,0]) and (perform_engarde_positioning == True):
        engarde_offset = engarde_offset + offset_increment
        display(f'The Engarde Positioning Failed, incrementing the Engarde Length by {offset_increment} frames.')
        if verbose == True:
          display(f'The Left Torso Position at the end of Engarde Positioning is: {Left_Torso_Position}.')
          display(f'The Right Torso Position at the end of Engarde Positioning is: {Right_Torso_Position}.')
      else:
        perform_engarde_positioning = False
        display(f'The Engarde Positioning was successful, continuing with clip analysis.')
        display(f'The final engarde length was {engarde_length + engarde_offset}.')
        if verbose == True:
          display(f'The Left Torso Position at the end of Engarde Positioning is: {Left_Torso_Position}.')
          display(f'The Right Torso Position at the end of Engarde Positioning is: {Right_Torso_Position}.')

    if verbose == True:
      display(f'The frame count is {frame_count} and perform engarde positioning is {perform_engarde_positioning}.')
    # Continues on the frames beyond the engarde length if positioning was successful
    if verbose == True:
      display(f'The frame count is {frame_count} and the engarde length is {engarde_length} with an offset of {engarde_offset}. Positioning is {perform_engarde_positioning}.')
    
    
    if (frame_count > (engarde_length + engarde_offset)) and (perform_engarde_positioning == False):

      positions = [Left_Position, Right_Position, Scoring_Box_Position, Left_Torso_Position, Right_Torso_Position, Left_Foot_Position, Right_Foot_Position]

      # Sets the certainty following the engarde positioning
      if frame_count == engarde_length + engarde_offset + 1:
        t3 = time.time()
        certainty = [0,0,0,0,0,0,0]
        if verbose == True:
          display(f'The positions following the engarde positioning are:')
          display(positions)

      if verbose == True:
        display(f'Certainty just prior to Bell Guard Positioning is {certainty}.')

      previous_certainty = certainty

      if right_torso_size_average[0] == 0 and left_torso_size_average[0] == 0:
        if verbose == True:
          display(f'Error, both Torso sizes are zero.')   
      elif right_torso_size_average[0] == 0:
        if verbose == True:
          display(f'Right Torso Size was zero, using the Left as a Defualt.')
        right_torso_size_average = left_torso_size_average
      elif left_torso_size_average[0] == 0:
        if verbose == True:
          display(f'Left Torso Size was zero, using the Right as a Defualt.')
        left_torso_size_average = right_torso_size_average

      if verbose == True:
        display(f'The engarde_length is {engarde_length}.')
        display(f'The engarde_offset is {engarde_offset}.')

      if verbose == True:
        display(f'Right Torso Size average is {right_torso_size_average} at frame {frame_count} just prior to Bell Guard Positioning.')
        display(f'Right Torso Position is {positions[4]} at frame {frame_count}.')

      # Finds the Tracked Items and Returns their positions
      [Left_Position_Temp, Right_Position_Temp, Scoring_Box_Position_Temp, Tracking_Bounding_Boxes_Temp, \
      Left_Torso_Position_Temp, Right_Torso_Position_Temp, Left_Foot_Position_Temp, Right_Foot_Position_Temp, certainty] \
      = Bell_Guard_Position_Finding(bbox, width, height, positions, frame_count, \
      left_torso_size_average, right_torso_size_average, (engarde_length + engarde_offset), \
      certainty, camera_steady, camera_motion_threshold, Exclusion_Areas, orig_img_worpt_starting_list)

      if verbose == True:
        display(f'Certainty just after to Bell Guard Positioning is {certainty}.')
        display(f'The Left Position at frame {frame_count - 1} is {Left_Position_Temp}.')

      if close_bellguards == True:
        frames_after_light_iterator = frames_after_light_iterator + 1

      # If the Positions are Close enough then the lights are assumed on and tracking is stopped
      if ((Right_Position_Temp[0] - Left_Position_Temp[0]) < width*position_difference_ratio) or (close_bellguards == True):
        close_bellguards = True
        # display(f'close_bellguards set to True')
        if frames_after_light_iterator == 0:
          display(f'The last tracked frame is {frame_count}.')
          # Adjusts the frames after light to include the entire clip
          if run_entire_video == True:
            frames_after_light_max = total_frames - frame_count
          # Ensures that frames processed does not exceed total number of frames
          if (frames_after_light_max + frame_count + starting_frame) >= total_frames:
            frames_after_light_max = total_frames - (frames_after_light_max + frame_count + starting_frame)
            display(f'The frames after light max has been adjusted to {frames_after_light_max}')

      # Appends the Returned Positions
      Left_Position.append(Left_Position_Temp)
      Right_Position.append(Right_Position_Temp)
      Scoring_Box_Position.append(Scoring_Box_Position_Temp)
      Left_Torso_Position.append(Left_Torso_Position_Temp)
      Right_Torso_Position.append(Right_Torso_Position_Temp)
      Left_Foot_Position.append(Left_Foot_Position_Temp)
      Right_Foot_Position.append(Right_Foot_Position_Temp)
      Tracking_Bounding_Boxes.append(Tracking_Bounding_Boxes_Temp)

      # Tests for a change in certainty to zero from non-zero. If a position has become certain during
      # this frame then it back calculates previous uncertain position up to a certain position.
      if (certainty[0] == 0 and previous_certainty[0] != 0):
        Left_Position = position_linear_approximation(Left_Position, previous_certainty[0])
        if verbose == True:
          display(f'Using a Linear Approximation for frame {frame_count} for the Left Bellguard Position.')
      elif (certainty[1] == 0 and previous_certainty[1] != 0):
        if verbose == True:
          display(f'The Right Position is: {Right_Position}.')
          display(f'The Previous Certainty is: {previous_certainty[1]}')
          display(f'Using a Linear Approximation for frame {frame_count} for the Right Bellguard Position.')
        Right_Position = position_linear_approximation(Right_Position, previous_certainty[1])
      elif (certainty[2] == 0 and previous_certainty[2] != 0):
        Scoring_Box_Position = position_linear_approximation(Scoring_Box_Position, previous_certainty[2])
      elif (certainty[3] == 0 and previous_certainty[3] != 0):
        Left_Torso_Position = position_linear_approximation(Left_Torso_Position, previous_certainty[3])
      elif (certainty[4] == 0 and previous_certainty[4] != 0):
        Right_Torso_Position = position_linear_approximation(Right_Torso_Position, previous_certainty[4])
      elif (certainty[5] == 0 and previous_certainty[5] != 0):
        Left_Foot_Position = position_linear_approximation(Left_Foot_Position, previous_certainty[5])
      elif (certainty[6] == 0 and previous_certainty[6] != 0):
        Right_Foot_Position = position_linear_approximation(Right_Foot_Position, previous_certainty[6])
      else:
        pass    

  # End of Individual Frame Processing

  # Goes through Position and Verifies that the scorebox images contain the scorebox position
  if use_scorebox_lights == True:
    default_scorebox_img_list = []
    default_scorebox_shape_width_list = []
    default_scorebox_shape_height_list = []
    for i in range(len(scorebox_bbox_list)):
      # Gets the Array that represents scorebox image
      scorebox_img = export_scorebox(orig_img_worpt_starting_list[i], scorebox_bbox_list[i], Scoring_Box_Position[i])
      if verbose_det:
        display(f'scorebox_bbox_list[i] is:')
        display(scorebox_bbox_list[i])
        display(f'Scoring_Box_Position[i] at frame {i} is:')
        display(Scoring_Box_Position[i])
      if i > engarde_length and len(scorebox_img) == 0:
        # If the scorebox_image is not detected and i is greater than engarde length, this may be becaue
        # there was no bbox detection so a default is used.
        bbox_temp = [int(Scoring_Box_Position[i][1] - scorebox_def_height/2), int(Scoring_Box_Position[i][0] - scorebox_def_width/2), int(Scoring_Box_Position[i][1] + scorebox_def_height/2), int(Scoring_Box_Position[i][0] + scorebox_def_width/2)]
        scorebox_bbox_list_default = [np.array(bbox_temp)]
        # [y_top, x_left, y_bottom,x_right]
        if verbose_det:
          display(f'Using default scorebox bbox {bbox_temp}.')
        scorebox_img = export_scorebox(orig_img_worpt_starting_list[i], scorebox_bbox_list_default, Scoring_Box_Position[i])
      # Use the ScoreBox Image Array to determine if the ScoreBox Lights are on
      # Returns [0/1,0/1] for on or off of Red, Green
      if i < engarde_length:
        if scorebox_img != []:
          default_scorebox_shape_height_list.append(scorebox_img.shape[0])
          default_scorebox_shape_width_list.append(scorebox_img.shape[1])
        default_scorebox_img_list.append(scorebox_img)
        # Assumes the lights are off during the engarde positioning
        ScoreBox_Lights_temp = [0,0]
      elif i == engarde_length and len(scorebox_img) > 0 and default_scorebox_img_list[0] != []:
        # img is a list of arrays that represent the scorebox images for the engarde positioning
        [ScoreBox_Default_Green, ScoreBox_Default_Red, image_size_default] = Find_ScoreBox_Default(default_scorebox_img_list)
        # Finds the scorebox default average height and width
        if default_scorebox_shape_height_list != []:
          scorebox_def_height = int(sum(default_scorebox_shape_height_list)/len(default_scorebox_shape_height_list))
          scorebox_def_width = int(sum(default_scorebox_shape_width_list)/len(default_scorebox_shape_width_list))
        else:
          # Creates a default, default scorebox size based on capture size.
          scorebox_def_height = int(height/8)
          scorebox_def_width = int(width/6)
      elif i > engarde_length and len(scorebox_img) > 0:
        ScoreBox_Lights_temp = Analyze_ScoreBox_Lights(file_name, ScoreBox_Default_Green, ScoreBox_Default_Red, image_size_default, scorebox_img)
      else:
        # ScoreBox_Lights.append([0,0])
        if verbose_det:
          display(f'The length of scorebox img is {len(scorebox_img)}.')
        display(f'Failed to detect images for the default scoring box, breaking the Light Detection Loop at frame {i}.')
        break
      ScoreBox_Lights.append(ScoreBox_Lights_temp)

    # display(f'The scorebox lights are:')
    # display(ScoreBox_Lights)
      
  if not too_few_frames:

    if verbose == True:
      # Reduces the Frame Count to account for skipped frames
      display(f'The original frame count was: {frame_count - 1} and the number of frames skipped is: {number_of_frames_skipped}.')
    frame_count = len(bbox)
    if verbose == True:
      display(f'The length of the frame_count is {frame_count - 1} while the number of bboxes is {len(bbox)}.')

    file_to_remove = r'/Mask_RCNN/videos/' + file_name
    try:
      shutil.rmtree(file_to_remove)
    except:
      if verbose == True:
        display(f'ERROR removing the video file to analyze.')

    if verbose == True:
      display(f'The Left Position just prior to drawing the Bell_Guards is:')
      display(Left_Position)
      display(f'The Right Position just prior to drawing the Bell_Guards is:')
      display(Right_Position)
      display(f'The Score Box Position just prior to drawing the Bell_Guards is:')
      display(Scoring_Box_Position)

    # Allows for a simple clip vector to draw in the lights if an overlay is used instead of a representative clip
    simple_clip_vector = simple_clip_vector_generator(file_name, Left_Position, Right_Position, width, ScoreBox_Lights)

    if verbose == True:
      display(f'The Right Torso ')
      display(f'The Right Torso Position Just Prior to Drawing at frame {frame_count} is: {Right_Torso_Position}')

    #Draws the Boxes on the image frame and determines scoring lights turned on
    [left_light_comparison, right_light_comparison, save_image_list] = \
    draw_Bell_Guard_Position(Left_Position, Right_Position, Scoring_Box_Position, scoring_box_size_average, \
    Left_Torso_Position, Right_Torso_Position, Left_Foot_Position, Right_Foot_Position, frame_count, Tracking_Bounding_Boxes, \
    width, height, engarde_length + engarde_offset, score_box_empty, camera_steady, camera_motion_threshold, \
    Exclusion_Areas, simple_clip_vector, save_image_list)


    if camera_motion_compensate == True and score_box_empty == False:
      #Adjusts the Bellguard Position Based on the Camera motion as determined by the Score_Box Position
      Left_Position = camera_motion_adjustment(Left_Position, Scoring_Box_Position)
      Right_Position = camera_motion_adjustment(Right_Position, Scoring_Box_Position)

    if camera_motion_compensate == True and score_box_empty == False:
      #Adjusts Left and Right Position for convenient visualization
      [Left_Position, Right_Position] = position_down_scale(Left_Position, Right_Position, width, height)

    clip_vector = clip_vector_generator(Left_Position, Right_Position, left_light_comparison, right_light_comparison, clip_vector_previous, width)

    if smooth_video_clip == True:
      #Smoothes the Clip using Savitzky Golay filter
      clip_vector = smooth_clip_vector(clip_vector, engarde_length + engarde_offset)

    if verbose == True:
      display(f'file_name[:-4] is {file_name[:-4]}.')
      # display(f'file_name[:-5] is {file_name[:-5]}.')
      display(f'Touch Folder is {touch_folder}.')
      display(f'File is {file}.')
      display(f'The final clip vector is:')
      display(clip_vector)

    clip_vector_np_save(touch_folder, file_name[:-4], clip_vector, file)

  else:
    # Sets output variables to none since the clip cannot run due to too few frames
    display(f'The clip failed to run. Clip Vector set to None.')
    bbox = clip_vector = 'none'

  return (bbox_clip, frame_count, width, height, clip_vector_previous, clip_vector, fps, save_image_list, too_few_frames)
