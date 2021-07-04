def Left_Right_Test(Left_Position, Right_Position):
  # Requires that the Left and Right BellGuards be on the Left and Right sides respectively

  #Left_Position is chosen arbitrarily for length
  for i in range(len(Left_Position)):
    if Left_Position[i][0] > Right_Position[i][0]:
      if verbose == True:
        display(f'The Left and Right were swapped on frame {i} and are now corrected.')
      position_temp = Left_Position[i]
      Left_Position[i] = Right_Position[i]
      Right_Position[i] = position_temp
    else:
      pass

  return (Left_Position, Right_Position)