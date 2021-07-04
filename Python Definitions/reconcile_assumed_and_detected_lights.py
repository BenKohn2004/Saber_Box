def reconcile_assumed_and_detected_lights(a,b):
  # Reconciling Light Lists
  # a is the assumed lights and b is the detected lights

  # Find Transition Point of Assumed Lights
  transition_radius = 5

  # Finds the transition point for the assumed lights
  for i in range(len(a)):
    if a[i][0] == 1:
      break

  # The assumed transition point is the first frame where the assumed lights are on
  assumed_transition = i - 1

  transition_down = transition_radius
  # Truncates the transition radius with the lights on if the clip doesn't contain enough frames
  transition_up = min(transition_radius, len(b) - i)

  if verbose_det == True:
    display(f'The transition point for the assumed lights is {assumed_transition}.')
    display(f'The distance is from -{transition_down} to {transition_up}.')

  # Crops the detected lights to the transition point from the assumed
  b_transition_range = b[i-transition_radius:i+transition_up]

  # Initializes the transition consistent to false
  b_transition_consistent = False

  # Checks for transition
  b_transition_sum = 0

  # Sums all the on and off values, effectively counting them
  for i in range(len(b_transition_range)):
    b_transition_sum = b_transition_sum + b_transition_range[i][0] + b_transition_range[i][1]

  # Tests if there is a sufficient number of both off and on lights in the transition range
  if b_transition_sum > 0.2 * len(b_transition_range) and b_transition_sum < 1.8 * len(b_transition_range):
    b_transition_consistent = True

    # Determine Start of Red and Green Lights

  if b_transition_consistent:
    # Cycles from the end to where the ligts are off
    for i in range(len(b_transition_range)-1,0,-1):
      if b_transition_range[i][0] == 0:
        break

    # Tests the consistency of Red Range
    red_off_total = 0
    red_on_total = 0
    red_consistent_threshold = 0.6
    for j in range(len(b_transition_range)):
      if j <= i:
        if verbose_det == True:
          display(f'Frame {assumed_transition + j}, {j} is off.')
        red_off_total = red_off_total + b_transition_range[j][0]
      else:
        if verbose_det == True:
          display(f'Frame {assumed_transition + j}, {j} is on.')
        red_on_total = red_on_total + b_transition_range[j][0]

    # The point within the total detection clip where red changes from off to on
    # The transition point is the last off light
    red_detection_transition = assumed_transition + i - transition_radius + 2

    if len(b_transition_range)-i-1 == 0:
      on_normalizer = 1
    else:
      on_normalizer = len(b_transition_range)-i-1

    if verbose_det == True:
      display(f'The Red off total/average is {red_off_total},{1-red_off_total/(i)} and Red on total/average is {red_on_total},{red_on_total/(on_normalizer)}')

    # if ((1-red_off_total/(i)) + (red_on_total/(on_normalizer)))/2 >= red_consistent_threshold:
    #   display(f'The red lights are consistent')
    #   red_consistent = True
    # else:
    #   display(f'The red lights are NOT consistent')
    #   red_consistent = False
    red_consistent = True


    for i in range(len(b_transition_range)-1,0,-1):
      if b_transition_range[i][1] == 0:
        break

    # Tests the consistency of Green Range
    green_off_total = 0
    green_on_total = 0
    green_consistent_threshold = 0.6
    for j in range(len(b_transition_range)):
      if j <= i:
        if verbose_det == True:
          display(f'Frame {assumed_transition + j}, {j} is off.')
        green_off_total = green_off_total + b_transition_range[j][1]
      else:
        if verbose_det == True:
          display(f'Frame {assumed_transition + j}, {j} is on.')
        green_on_total = green_on_total + b_transition_range[j][1]

    if verbose_det == True:
      display(f'The Green off total/average is {green_off_total},{1-green_off_total/(i)} and Green on total/average is {green_on_total},{green_on_total/(on_normalizer)}')
      display(f'The transition for green detected is {i}.')

    # The point within the total detection clip where green changes from off to on
    green_detection_transition = assumed_transition + i - transition_radius + 2

    # if ((1-green_off_total/(i)) + (green_on_total/(on_normalizer)))/2 >= green_consistent_threshold:
    #   display(f'The green lights are consistent')
    #   green_consistent = True
    # else:
    #   display(f'The green lights are NOT consistent')
    #   green_consistent = False
    green_consistent = True

    if verbose_det == True:
      display(f'The red/green transition points are {red_detection_transition},{green_detection_transition}.')
      display(f'Red Consistent is {red_consistent}.')
      display(f'Green Consistent is {green_consistent}.')
  else:
    red_consistent = False
    green_consistent = False

  # Recreates the Detected Lights removing inconsistencies
  if red_consistent == True and green_consistent == True:
    light_vector_detection = []
    for i in range(len(a)):
      light_vector_detection_temp = []
      if i < red_detection_transition:
        light_vector_detection_temp.append(0)
      else:
        light_vector_detection_temp.append(1)
      if i < green_detection_transition:
        light_vector_detection_temp.append(0)
      else:
        light_vector_detection_temp.append(1)
      light_vector_detection.append(light_vector_detection_temp)
    consistent = True
  else:
    consistent = False
    light_vector_detection = []

  if verbose_det == True:
    display(f'The comparison of the assumed and detected lights are:')
    for i in range(len(light_vector_detection)):
      display(f'{i}, [{light_vector_detection[i][0]},{light_vector_detection[i][1]}]')

  return (light_vector_detection, consistent)