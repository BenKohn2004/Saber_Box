def simple_clip_vector_generator(file_name, Left_Position, Right_Position, width, light_vector_detection):
  #Compiles the clip_vector that is used for the action analysis

  # Allows for the assumption that both lights are on if the positions are close to each other.
  # Useful if there is difficulty detecting the scoring box.
  close_bellguards = False
  # Once lights turn on it is assumed the lights stay on for the rest of the action
  light_assumption = False

  if verbose == True:
    display(f'The Left Position is {Left_Position}.')
    display(f'The Right Position is {Right_Position}.')

  if len(Left_Position) != len(Right_Position):
    display(f'The Left and Right Positions do not match up')
  else:
    pass

  # Creates the Clip Vector
  clip_vector = []
  # Creates a Light Vector that is the last two columns of the clip vector so that it can be analyzed with Light Detection
  light_vector_assumed = []

  for i in range(len(Left_Position)):  
    # Checks the lights should be assumed on if they are not already
    # Determines if the bellguards are close to each other
    if ((Right_Position[i][0] - Left_Position[i][0]) < width*position_difference_ratio) and (light_assumption == False):
      close_bellguards = True

    # Adjusts the clip vector to reflect scoring box light assumptions
    clip_vector_temp = [[],[],[],[]]
    light_vector_assumed_temp = [[],[]]
    clip_vector_temp[0] = Left_Position[i][0]
    clip_vector_temp[1] = Right_Position[i][0]
    if (assume_lights == True and close_bellguards == True) or light_assumption == True:
      clip_vector_temp[2] = 1
      clip_vector_temp[3] = 1
      light_vector_assumed_temp[0] = light_vector_assumed_temp[1] = 1
      light_assumption = True
      # Updates the Lights on the ScoreBox DataFrame csv
      if export_scorebox_image == True:
        update_scorebox_csv(file_name, i+1, 'On')
    else:
      clip_vector_temp[2] = 0
      clip_vector_temp[3] = 0
      light_vector_assumed_temp[0] = light_vector_assumed_temp[1] = 0
      # Updates the Lights on the ScoreBox DataFrame csv
      if export_scorebox_image == True:
        update_scorebox_csv(file_name, i+1, 'Off')

    clip_vector.append(clip_vector_temp)
    light_vector_assumed.append(light_vector_assumed_temp)

  # display(light_vector_assumed)
  # display(light_vector_detection)

  # Skips the light comparison if it failed to detect Lights Prior
  if len(light_vector_assumed) == len(light_vector_detection):
    # display(f'The comparison of the assumed and detected lights are:')
    # for i in range(len(light_vector_assumed)):
      # display(f'{i}, [{light_vector_assumed[i][0]},{light_vector_assumed[i][1]}],[{light_vector_detection[i][0]},{light_vector_detection[i][1]}]')

    [light_vector_detection, light_consistency] = reconcile_assumed_and_detected_lights(light_vector_assumed,light_vector_detection)
  else:
    light_consistency = False


  # If the light vector detections are consistent then it replaces the assumed lights.
  if light_consistency:
    for i in range(len(clip_vector)):
      clip_vector[i][2] = light_vector_detection[i][0]
      clip_vector[i][3] = light_vector_detection[i][1]
    display(f'Updated the Light Vector to Detection.')
    if verbose_det:
      display(f'The Updated Clip Vector is:')
      for i in range(len(clip_vector)):
        display(f'Frame {i}, [{clip_vector[i][0]},{clip_vector[i][1]},{clip_vector[i][2]},{clip_vector[i][3]}]')

  return (clip_vector)