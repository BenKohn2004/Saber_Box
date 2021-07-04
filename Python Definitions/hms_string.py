def hms_string(sec_elapsed):
  # Nicely formatted time string
  h = int(sec_elapsed / (60 * 60))
  m = int((sec_elapsed % (60 * 60)) / 60)
  s = int(sec_elapsed % 60)
  ms = int(sec_elapsed * 1000 % 1000)