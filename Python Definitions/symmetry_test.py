def symmetry_test(width, height, left_x, left_y, right_x, right_y):

  # Tests the potential left and right positions for left/right symmetry and removes outlier points
  display(f'Commencing Symmetry Test...')

  # Sets how large the allowable band is with respect to height or width
  band_width_ratio_x = 8
  band_width_ratio_y = 8

  all_positions_x = left_x + right_x
  all_positions_y = left_y +right_y
  if len(all_positions_x) != len(all_positions_y):
    display(f'ERROR...The length of the x and y positions are different.')

  # Keeps track of which positions are most in line with the other positions
  # Finds the X Band
  x_distances_from_center = []
  x_distances_from_other_points_score = []
  for i in range(len(all_positions_x)):
    #Determines the x_min band for each position by distance from center
    x_distances_from_center.append(abs(int((width/2)-all_positions_x[i])))
  #Creates an iterator that determines which x_point is close to the most other points and finds its index
  for j in range(len(x_distances_from_center)):
    score = 0
    for k in range(len(x_distances_from_center) - 1):
      if abs(x_distances_from_center[j] - x_distances_from_center[k+1]) < width/band_width_ratio_x:
        score = score + 1
      else:
        pass
    x_distances_from_other_points_score.append(score)
  x_index_band = x_distances_from_other_points_score.index(max(x_distances_from_other_points_score))

  x_min = abs(int(all_positions_x[x_index_band] - width/band_width_ratio_x))
  x_max = abs(int(all_positions_x[x_index_band] + width/band_width_ratio_x))

  # Finds the Y Band
  y_distances_from_center = []
  y_distances_from_other_points_score = []
  for i in range(len(all_positions_y)):
    y_distances_from_center.append(abs(int((height/2)-all_positions_y[i])))
  for j in range(len(y_distances_from_center)):
    score = 0
    for k in range(len(y_distances_from_center) - 1):
      if abs(y_distances_from_center[j] - y_distances_from_center[k+1]) < width/band_width_ratio_y:
        score = score + 1
      else:
        pass
    y_distances_from_other_points_score.append(score)
  y_index_band = y_distances_from_other_points_score.index(max(y_distances_from_other_points_score))

  y_min = abs(int(all_positions_y[y_index_band] - width/band_width_ratio_y))
  y_max = abs(int(all_positions_y[y_index_band] + width/band_width_ratio_y))

  # Cycles through the positions and keeps values that are in the horizontal x band
  positionsx_temp = []
  positionsy_temp = []

  if verbose == True:
    display(f'The x_min/max is {x_min}/{x_max}, the band width is {width/band_width_ratio_x} and the center is {width/2}.')

  for i in range(len(all_positions_x)):
    if ((all_positions_x[i] < (width/2 - x_min)) and (all_positions_x[i] > (width/2 - x_max))) or ((all_positions_x[i] < (width/2 + x_max)) and (all_positions_x[i] > (width/2 + x_min))):
      positionsx_temp.append(all_positions_x[i])
      positionsy_temp.append(all_positions_y[i])
    else:
      pass

  # Replaces the all position x and y lists with the temp list limited by the bands
  all_positions_x = positionsx_temp
  all_positions_y = positionsy_temp

  #Cycles through the positions and keeps values that are in the vertical y band
  positionsx_temp = []
  positionsy_temp = []

  if verbose == True:
    display(f'The y_min/max is {y_min}/{y_max}, the band width is {height/band_width_ratio_y} and the center is {height/2}.')

  for i in range(len(all_positions_y)):
    if ((all_positions_y[i] > (y_min)) and (all_positions_y[i] < (y_max))):
      positionsx_temp.append(all_positions_x[i])
      positionsy_temp.append(all_positions_y[i])
    else:
      pass

  # Replaces the all position x and y lists with the temp list limited by the bands
  all_positions_x = positionsx_temp
  all_positions_y = positionsy_temp

  if verbose == True:
    display(f'There were originaly {len(left_x) + len(right_x)} values and {len(all_positions_x) - (len(left_x) + len(right_x))} were removed.')

  # Returns the x and y values to left and right positions
  ret_left_x, ret_left_y, ret_right_x, ret_right_y = [],[],[],[]

  for i in range(len(all_positions_x)):
    # Tests if the x value is on the left or right side
    if all_positions_x[i] < width/2:
      ret_left_x.append(all_positions_x[i])
      ret_left_y.append(all_positions_y[i])
    else:
      ret_right_x.append(all_positions_x[i])
      ret_right_y.append(all_positions_y[i])
  # Prevents an off center camera from removing all engarde points
  if (len(ret_left_x) == 0) or (len(ret_left_y) == 0) or (len(ret_right_x) == 0) or (len(ret_right_y) == 0):
    ret_left_x = left_x
    ret_left_y = left_y
    ret_right_x = right_x
    ret_right_y = right_y

  return (ret_left_x, ret_left_y, ret_right_x, ret_right_y)