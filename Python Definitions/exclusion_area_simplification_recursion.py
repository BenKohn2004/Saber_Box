def exclusion_area_simplification_recursion(a, max_dist):
  a_simplified = exclusion_area_simplification(a, max_dist)
  a_simplified_temp = 0

  while a_simplified != a_simplified_temp:
    if verbose == True:
      display(a_simplified)
    a_simplified_temp = exclusion_area_simplification(a_simplified, max_dist)
    if a_simplified_temp != a_simplified:
      a_simplified = a_simplified_temp
      a_simplified_temp = 0

  if a_simplified_temp == a_simplified:
    if verbose == True:
      display(f'The two are equal.')
  else:
    if verbose == True:
      display(f'The two are NOT equal.')

  return (a_simplified)