def Find_Array_Max_Value(ScoreBox_Default_Green, histr_green, ScoreBox_Default_Red, histr_red):

  array_max_value = 0

  for i in range(len(histr_green)):
    array_max_value = max(array_max_value, max(ScoreBox_Default_Green[i]))
    array_max_value = max(array_max_value, max(histr_green[i]))
    array_max_value = max(array_max_value, max(ScoreBox_Default_Red[i]))
    array_max_value = max(array_max_value, max(histr_red[i]))

  return (array_max_value)