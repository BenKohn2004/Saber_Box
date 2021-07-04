def downsample_fps(a,b):
  # Adjusts the elements of a larger set a to fit into the length of set b

  c = []
  remainder = 0
  for i in range(len(b)):
    c_temp = []
    if verbose == True:
      display(f'The lower range is {math.ceil(len(a)/len(b)*(i+1)-1-remainder)} and the upper range is {math.floor(len(a)/len(b)*(i+1))}.')
    for j in range(math.ceil(len(a)/len(b)*(i)-remainder),math.floor(len(a)/len(b)*(i+1))):
      c_temp.append(a[j])
      if verbose == True:
        display(f'i,j = {i},{j} and c_temp = {c_temp}')
    remainder = (len(a)/len(b))*(i+1) - int(len(a)/len(b)*(i+1))
    c.append(round(sum(c_temp)/len(c_temp)))
    if verbose == True:
      display(f'The remainder at i = {i} and j = {j} is {remainder} and c is {c}.')

  return (c)