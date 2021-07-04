def create_difference_array(array1, array2, max_value):

  difference_histogram_array =[[],[],[]]
  for i in range(3):
    difference_histogram_array[i] = ((array1[i] - array2[i])/max_value)

  return (difference_histogram_array)