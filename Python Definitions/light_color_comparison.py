def light_color_comparison(light_color, light_color_default, color):
  # Deterines if a light turned on based on a default color, an input color and expected color

  light_comparison = []
  # A high max distance is less sensitive and a lower max distance is more sensitive
  max_distance_total = 180
  max_distance_specific_color = 90

  if color == 'Red':
    color_specific = 2
  elif color == 'Green':
    color_specific = 1
  else:
    pass

  if verbose == True:
    display(f'The Color being analyzed is {color}.')
    display(f'The default color is:')
    display(light_color_default)
    display(f'With the specific color being {light_color_default[color_specific]}')
    display(f'The max distance total is {max_distance_total}.')
    display(f'The max distance for a specific color is {max_distance_specific_color}.')

  #i cycles through each light value corresponding to each frame
  for i in range(len(light_color)):
    distance = 0
    for j in range(3):
      distance = distance + (light_color[i][j] - light_color_default[j])**2

    distance_specific_color = abs(light_color[i][color_specific] - light_color_default[color_specific])

    distance = int((distance)**(0.5))
    if vebose == True:
      display(f'The distance is {distance} and the color specific distance is {distance_specific_color} for frame {i}.')
    #0 is no color change from the default color)
    if (distance > max_distance_total and distance_specific_color > max_distance_specific_color):
      light_comparison.append(1)
      if verbose == True:
        display(f'The light is ON.')
    #1 is a color change from the default color
    else:
      light_comparison.append(0)
      if verbose == True:
        display(f'The light is OFF.')

  return (light_comparison)