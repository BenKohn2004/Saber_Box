def exclusion_area_simplification(a, max_dist):
  a_simplified = []
  for i in range(len(a)):
    for j in range(len(a) - (i+1)):
      # a[i] - a[(i+1)+j]
      if a[i] != 'skip' and a[(i+1)+j] != 'skip':
        dist = int(((a[i][0] - a[(i+1)+j][0])**2 + (a[i][1] - a[(i+1)+j][1])**2)**(0.5))
        # display(f'The distance between {a[i]} and {a[(i+1)+j]} is {dist}.')
        if dist < max_dist:
          a_simplified.append(a[i])
          a[i] = 'skip'
          a[(i+1)+j] = 'skip'

  for k in range(len(a)):
    if a[k] != 'skip':
      a_simplified.append(a[k])

  return (a_simplified)