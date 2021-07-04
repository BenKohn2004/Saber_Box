def multiple_box_determination(expected_position, positions, x_boundaries, min_conf, horiz_flip):

  confidence_weighting = .9

  delta_x_forward = x_boundaries[1]
  delta_x_backward = x_boundaries[0]

  if horiz_flip == True:
    delta_temp = delta_x_forward
    delta_x_forward = delta_x_backward
    delta_x_backward = delta_temp

  position_ratings = []

  if verbose == True:
    display(f'There are {len(positions)} positions available.')
    display(f'The positions are:')
    display(positions)  

  for i in range(len(positions)):
    delta_position = positions[i][0] - expected_position[0]
    if verbose == True:
      display(f'The positions{i}[0] is {positions[i][0]} and the expected_position[0] is {expected_position[0]} therefore delta position is {delta_position}.')
    if delta_position > 0:
      if verbose == True:
        display(f'Position {i} is forward of the expected position.')
      position_ratings.append(abs((delta_position/delta_x_forward)*(1-positions[i][2])**confidence_weighting))
      if verbose == True:
        display(f'delta_position is {delta_position}.')
        display(f'delta_x_forward is {delta_x_forward}.')
        display(f'positions[i][2] is {positions[i][2]}.')
    else:
      if verbose == True:
        display(f'Position {i} is behind the expected position.')
      position_ratings.append(abs((delta_position/delta_x_backward)*(1-positions[i][2])**confidence_weighting))
      if verbose == True:
        display(f'delta_position is {delta_position}.')
        display(f'delta_x_backward is {delta_x_backward}.')
        display(f'positions[i][2] is {positions[i][2]}.')

  if verbose == True:
    display(position_ratings)

  position = positions[position_ratings.index(min(position_ratings))]

  return (position)