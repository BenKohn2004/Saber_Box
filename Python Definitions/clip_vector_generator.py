def clip_vector_generator(Left_Position, Right_Position, left_light_comparison, right_light_comparison, clip_vector_previous, width):
  #Compiles the clip_vector that is used for the action analysis

  # Allows for the assumption that both lights are on if the positions are close to each other.
  # Useful if there is difficulty detecting the scoring box.
  close_bellguards = False
  # Once lights turn on it is assumed the lights stay on for the rest of the action
  light_assumption = False

  if len(Left_Position) != len(Right_Position):
    display(f'The Left and Right Positions do not match up')
  else:
    pass

  # This is either [] or the Previously saved Clip_Vector
  clip_vector = clip_vector_previous

  for i in range(len(Left_Position)):  
    # Checks if the lights should be assumed on if they are not already
    # Determines if the bellguards are close to each other
    # if (abs(Left_Position[i][0] - Right_Position[i][0]) < width*.050) and (light_assumption == False):
    if ((Right_Position[i][0] - Left_Position[i][0]) < width*position_difference_ratio) and (light_assumption == False):
      close_bellguards = True

    # Adjusts the clip vector to reflect scoring box light assumptions
    clip_vector_temp = [[],[],[],[]]
    clip_vector_temp[0] = Left_Position[i][0]
    clip_vector_temp[1] = Right_Position[i][0]
    if (assume_lights == True and close_bellguards == True) or light_assumption == True:
      clip_vector_temp[2] = 1
      clip_vector_temp[3] = 1
      light_assumption = True
    else:
      # if ignore_box_lights == True:
      clip_vector_temp[2] = 0
      clip_vector_temp[3] = 0
      # else:
      #   clip_vector_temp[2] = left_light_comparison[i]
      #   clip_vector_temp[3] = right_light_comparison[i]

    clip_vector.append(clip_vector_temp)

  return (clip_vector)