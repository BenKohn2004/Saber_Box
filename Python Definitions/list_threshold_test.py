def list_threshold_test(threshold, list_to_test):
  #Determines if a list meets a minimum threshold
  threshold_met = False

  for k in range(len(list_to_test)):
    if list_to_test[k][1] > threshold:
      threshold_met = True
      break
    else:
      pass

  return (threshold_met)