def average_list(List):
  # Finds the Average of a List
  try:
    average = sum(List) / len(List)
  except:
    average = 0
  return (average)