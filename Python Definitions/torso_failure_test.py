def torso_failure_test(bbox, capture_width, capture_height, y_average, Bell_Guard_Size_average, side, frame_count, min_torso_confidence):
  # Tests for reasons the engarde positioning failed to detect a Torso
  # Is tested at finding tracking boxes

  if verbose == True:
    display(f'The {side} Torso failed due to...') 
  for j in range(len(bbox)):
    if bbox[j][1] > min_torso_confidence:
      pass
    else:
      if verbose == True:
        display(f'The confidence is of the box is too low at only {int(bbox[j][1]*100)}% at frame {frame_count}.')
    if bbox[j][0][2] > y_average:
      pass
    else:
      if verbose == True:
        display(f'The Torso was not lower than the Bell Guard with a lower height of {bbox[j][0][2]} with a max value of {y_average} at frame {frame_count}.')
    if bbox[j][0][2] < (y_average + 3*Bell_Guard_Size_average[1]):
      pass
    else:
      if verbose == True:
        display(f'The bottom of the torso box was too low at {bbox[j][0][2]} with a max value of {int(y_average + 3*Bell_Guard_Size_average[1])} at frame {frame_count}.')

  if verbose == True:
    display(f'y_average is {y_average}.')
    display(f'Bell_Guard_Size_average[1] is {Bell_Guard_Size_average[1]}.')

  return