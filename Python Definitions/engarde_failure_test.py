def engarde_failure_test(bbox, bellguard_confidence, x_max, y_max, side):
  # Tests for reasons the engarde positioning failed to detect a BellGuard

  if verbose == True:
    display(f'The {side} engarde position failed due to...')

  if side == 'Left':
    oppside = 'Right'
    k = 0
  else:
    oppside = 'Left'
    k = 1

  for j in range(len(bbox)):
    if bbox[j][1] < bellguard_confidence:
      if verbose == True:
        display(f'The confidence in the {side} bellguard is too low at {bellguard_confidence}.')
    else: 
      pass
    if side == 'Left':
      if bbox[j][k] > x_max:
        if verbose == True:
          display(f'The {side} bellguard was too far {oppside} at {bbox[j][0]} while the maximum is {x_max}.')
      else:
        pass
    else:
      if verbose == True:
        display(f'bbox at this point is: {bbox}. J is {j} and k is {k}.')
        display(bbox[j])
        display(bbox[j][k])
      if bbox[j][k] < x_max:
        if verbose == True:
          display(f'The {side} bellguard was too far {oppside} at {bbox[j][0]} while the maximum is {x_max}.')
    if bbox[j][k] > y_max:
      if verbose == True:
        display(f'The {side} bellguard was too low at {bbox[j][0]} while the maximum allowed is {y_max}.')
    else:
      pass

  return