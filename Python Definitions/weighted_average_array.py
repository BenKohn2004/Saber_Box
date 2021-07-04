def weighted_average_array(arr):

  weighted_sum = 0

  for i in range(arr.shape[0]):
    weighted_sum = weighted_sum + i*arr[i]

  weighted_average = weighted_sum/arr.shape[0]

  return (weighted_average)