def average_list_without_null(List):
  # Takes a List of x,y values and null values. Removes the null values and returns the average x and y values.

  List_temp = []
  for i in range(len(List)):
    if List[i] != []:
      List_temp.append(List[i])
    else:
      pass
  
  x_sum = 0
  y_sum = 0

  # display(f'List_Temp is {List_temp}')

  for j in range(len(List_temp)):
    x_sum = x_sum + List_temp[j][0]
    y_sum = y_sum + List_temp[j][1]

  x_average = int(x_sum/len(List_temp))
  y_average = int(y_sum/len(List_temp))

  average = [x_average, y_average]

  return (average)