def load_clip_vector():
  # Only used for large clips

  display(f'Loading the Clip Vector...')
  filename = r'/content/drive/My Drive/projects/fencing/Fencing Clips/Temp_Clip_Vector/Temp_Clip_Vector_Clips/clip_vector_np1.csv'

  display(f'Attempting to load:')
  display(filename)
  try:
    vector_data = pd.read_csv(filename, header=None)
    arr = vector_data.to_numpy(dtype = np.int32)
    clip_vector = arr.tolist()
  except:
    display(f'Load Failure...')
    display(f'The clip_vector did not exist so it is set to []')
    clip_vector = []

  return (clip_vector)