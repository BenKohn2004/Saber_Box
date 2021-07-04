def torso_position_failure_test(bbox, engarde_length, x_min_torso, x_max_torso, y_min_torso, y_max_torso, y_average, side, frame_count):
  # Tests for reasons the engarde positioning failed to detect a Torso
  # Is tested at torso positions

  confidence = min_torso_confidence

  if verbose == True:
    display(f'Analyzing the Torso Position Failure at frame {frame_count} for the {side} side...')
  count = 0
  
  for k in range(len(bbox)):

    if bbox[k][2] == 3 and bbox[k][1] > confidence:
      count = count + 1
  if verbose == True:
    display(f'There are {len(bbox)} ROIs, {count} of them are Torsos with greater than {confidence}%.')

  for j in range(len(bbox)):
    y_center = int((bbox[j][0][0] + bbox[j][0][2])/2)
    x_center = int((bbox[j][0][1] + bbox[j][0][3])/2)
    if bbox[j][2] == 3 and bbox[j][1] > confidence:
      if x_center > x_min_torso:
        pass
      else:
        if verbose == True:
          display(f'The Torso center at {x_center} is to the Left of the Box side at {x_min_torso} at frame {frame_count}.')
      if x_center < x_max_torso:
        pass
      else:
        if verbose == True:
          display(f'The Torso center at {x_center} is to the Right of the Box side at {x_max_torso} at frame {frame_count}.')
      if y_center > y_min_torso:
        pass
      else:
        if verbose == True:
          display(f'The Torso center at {y_center} is Above the Box at {y_min_torso} at frame {frame_count}.')
      if y_center < y_max_torso:
        pass
      else:
        if verbose == True:
          display(f'The Torso center at {y_center} is Below the Box at {y_max_torso} at frame {frame_count}.')
      if bbox[j][0][2] > y_average:
        pass
      else:
        if verbose == True:
          display(f'The Torso center is Below the Bell Guard at frame {frame_count}.')
      if bbox[j][2] == 3:
        pass
      else:
        if verbose == True:
          display(f'The Torso is not labelled as a Torso at frame {frame_count}.')
    else:
      pass

  return