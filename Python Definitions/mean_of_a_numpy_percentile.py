def mean_of_a_numpy_percentile(arr, percentile_cutoff):
  # Returns a percentile value of a numpy array

  if verbose == True:
    display(f'The average of arr is {np.average(arr)}.')

  percentile_value = np.percentile(arr, percentile_cutoff)

  # Uses just the percentile without averaging
  array_percentile_mean = percentile_value

  return (array_percentile_mean)