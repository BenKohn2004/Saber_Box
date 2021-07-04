def load_clip(folder, file_name, capture_width):
  if folder == 'Left' or folder == 'left' or folder == 'Left_Touch':
    folder = 0
  if folder == 'Right' or folder == 'right'or folder == 'Right_Touch':
    folder = 1
  if folder == 'Simul' or folder == 'simul'or folder == 'Simul':
    folder = 2

  engarde_position_buffer = 15
  max_length = 111
  clip_vector_length = max_length - engarde_position_buffer


  touch_folder = ['Left_Touch', 'Right_Touch', 'Simul']

  i = folder

  # file = 'clip_vector_acceleration_np' + str(clip_number) + '.csv'
  file = file_name
  path = folder
  # path = r'/content/drive/My Drive/projects/fencing/Fencing Clips/' + touch_folder[i] + '/' + touch_folder[i] + '_Vector_Clips_Acceleration/'

  vector_data = pd.read_csv(os.path.join(path, file), header=None)
  clip_vector = vector_data.to_numpy(dtype = np.float32)

  display(os.path.join(path, file))

  # Pads the clip_vector to max_length
  # If the clip is greater than Max Length, it is truncated
  if len(clip_vector) > max_length:
    clip_vector = clip_vector[len(clip_vector) - max_length:]
  padding = np.array([0,0,0,0])
  for k in range(max_length - (len(clip_vector))):
    clip_vector = np.vstack((clip_vector, padding))

  #Normalizes the Values
  max_value = int(capture_width/42)
  for i in range(len(clip_vector)):
    for j in range(2):
      if clip_vector[i][j] < max_value:
        clip_vector[i][j] = clip_vector[i][j] * (1/max_value)
      else:
        #Preserves the sign of the value
        clip_vector[i][j] = clip_vector[i][j]/(abs(clip_vector[i][j]))

  # Removes the First 15 frames to minimize engarde positioning
  clip_vector = clip_vector[15:]

  # Sets Clip_Vector to Zero if Light is on
  for j in range(len(clip_vector)):
    if clip_vector[j][2] == 1:
      clip_vector[j][0] = 0
    if clip_vector[j][3] == 1:
      clip_vector[j][1] = 0 

  clip_vector = clip_vector.reshape(1,clip_vector_length,4)
  return (clip_vector)