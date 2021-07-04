def position_linear_approximation(position, previous_certainty):
  # Certainty is the number of times previous to current position that a point was not certain.
  last_known_position = ((previous_certainty+2)*(-1))

  # Finds the positional distance between two known boxes
  x_delta = int((position[-1][0] - position[last_known_position][0])/(last_known_position+1))
  y_delta = int((position[-1][1] - position[last_known_position][1])/(last_known_position+1))
  delta = [x_delta, y_delta]

  # Adjusts the previous positions, up to the previous certainty, based on a linear approximation
  for j in range(2):
    for i in range(previous_certainty+1):
      position[i - (previous_certainty+1)][j] = position[i - (previous_certainty+2)][j] - delta[j]

  return (position)