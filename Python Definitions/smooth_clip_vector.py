def smooth_clip_vector(clip_vector, engarde_length):
  # Allows for smoothing the clip_vector

  a = []
  b = []
  for i in range(engarde_length, len(clip_vector)):
    a.append(clip_vector[i][0])
    b.append(clip_vector[i][1])

  x = np.linspace(engarde_length,len(clip_vector), len(clip_vector) - engarde_length)

  # sos = signal.ellip(13, 0.009, 80, 0.05, output='sos')
  # yhata = signal.sosfilt(sos, a)
  if len(a)%2 == 1:
    yhata = signal.savgol_filter(a, len(a), 11)
    yhatb = signal.savgol_filter(b, len(b), 11)
  else:
    yhata = signal.savgol_filter(a, len(a) - 1, 11)
    yhatb = signal.savgol_filter(b, len(b) - 1, 11)    

  # plt.plot(x,a, color='black')
  # plt.plot(x,yhata, color='red')
  plt.plot(x,b, color='black')
  plt.plot(x,yhatb, color='blue')
  plt.show()

  vector_clip_smooth = []

  for j in range(len(clip_vector)):
    if j <= engarde_length:
      clip_vector_smooth_temp = [clip_vector[j][0], clip_vector[j][1], clip_vector[j][2], clip_vector[j][3]]
    else:
      clip_vector_smooth_temp = [int(yhata[j - engarde_length]), int(yhatb[j - engarde_length]), clip_vector[j][2], clip_vector[j][3]]
    vector_clip_smooth.append(clip_vector_smooth_temp)

  return (vector_clip_smooth)