def boundary_box_test(test_point, boundary):
  # Tests if a give point is in a Boundary Box.
  #Format Test_Point is of the form (x,y)
  #Format Boundary is of the form (x_min, x_max, y_min, y_max)
  #Format Boundary is of the form (behind the fencer, in front of the fencer, above the fencer, below the fencer)

  if verbose == True:
    display(test_point)
    display(boundary)

  if test_point != 'None':
    if test_point[0] > boundary[0] and test_point[0] < boundary[1] and test_point[1] > boundary[2] and test_point[1] < boundary[3]:
      box_test = True
    else:
      box_test = False
  else:
    box_test = False

  return (box_test)