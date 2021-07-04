def Find_Difference_And_Format(histr_green, ScoreBox_Default_Green, histr_red, ScoreBox_Default_Red, image_size, image_size_default):

  x_green = []
  x_red = []

  max_shape = int((max(image_size, image_size_default))/50)

  difference_array_Green = np.array(create_difference_array(histr_green, ScoreBox_Default_Green, max_shape))
  difference_array_Red = np.array(create_difference_array(histr_red, ScoreBox_Default_Red, max_shape))

  x_green.append(difference_array_Green)
  x_red.append(difference_array_Red)

  x_green_arr = np.stack(x_green, axis=0)
  x_red_arr = np.stack(x_red, axis=0)

  # Reshapes the 3 color (H L S) to a single 768 vector
  x_green_arr_1 = x_green_arr.reshape(x_green_arr.shape[0],768)
  x_red_arr_1 = x_red_arr.reshape(x_red_arr.shape[0],768)

  x_green = x_green_arr_1
  x_red = x_red_arr_1

  return (x_red, x_green)