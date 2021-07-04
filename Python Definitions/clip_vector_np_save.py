def clip_vector_np_save(clip_call, file_number, clip_vector, file_title):
  # Saves the clip vector for future use
  # Clip_Call Left_Touch, Right_Touch, Simul

  # Generates the clip_vector speed based on the clip_vector
  clip_vector_speed = []
  for i in range(len(clip_vector)-1):
    clip_vector_speed.append([])
    clip_vector_speed[i].append(clip_vector[i+1][0]-clip_vector[i][0])
    # Reverses the Right Fencers position so that positive is towards the opponent
    clip_vector_speed[i].append(clip_vector[i][1]-clip_vector[i+1][1])
    clip_vector_speed[i].append(clip_vector[i+1][2])
    clip_vector_speed[i].append(clip_vector[i+1][3])

  # Generates the clip_vector acceleration based on the clip_vector
  clip_vector_acceleration = []
  for i in range(len(clip_vector_speed)-1):
    clip_vector_acceleration.append([])
    clip_vector_acceleration[i].append(clip_vector[i+1][0]-clip_vector[i][0])
    # Reverses the Right Fencers position so that positive is towards the opponent
    clip_vector_acceleration[i].append(clip_vector[i][1]-clip_vector[i+1][1])
    clip_vector_acceleration[i].append(clip_vector[i+1][2])
    clip_vector_acceleration[i].append(clip_vector[i+1][3])

  # path = '/content/drive/My Drive/projects/fencing/Fencing Clips/'

  # # Saves the clip_vector_acceleration
  clip_vector_acceleration_np = np.asarray(clip_vector_acceleration)

  # Saves a Copy to the Sync Folder
  file_title = file_title[:-4]
  clip_vector_acceleration_np_name = file_title + '_acc.csv'
  sync_folder_acc = '/content/drive/My Drive/Sync/Acceleration'
  os.chdir(sync_folder_acc)
  np.savetxt(clip_vector_acceleration_np_name, clip_vector_acceleration_np, delimiter=',')

  return