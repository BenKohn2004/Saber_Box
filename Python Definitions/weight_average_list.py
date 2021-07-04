def weight_average_list(List):
  # Finds the Weight Average of a List

  # Prevents division by zero
  try:
    value_sum = 0
    value_weight = 0
    for i in range(len(List)):
      value_sum = value_sum + List[i][0] * List[i][1]
      value_weight = value_weight + List[i][1]
    weighted_average = value_sum/value_weight
  except:
    weighted_average = 0

  return (weighted_average)