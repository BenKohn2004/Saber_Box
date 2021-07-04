def create_boundary_box(center, padding, horiz_flip):
  # Creates a Boundary Box based on Center Padding and if the Left and Right Boundaries should be flipped.
  # Center is [x,y]
  # Padding is [Behind, Front, Top, Bottom]
  # horiz_flip is True or False

  if horiz_flip == False:
    left = center[0] - padding[0]
    right = center[0] + padding[1]
  elif horiz_flip == True:
    left = center[0] - padding[1]
    right = center[0] + padding[0]
  else:
    if verbose == True:
      display(f'ERROR Horiz Flip not True or False.')

  top = center[1] - padding[2]
  bottom = center[1] + padding[3]

  return ([left, right, top, bottom])