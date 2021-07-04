def run_human_pose(image, fencer_data, keypoints, already_run):
  # Runs Human Pose Analysis on an Image and returns fencer data and keypoints
  # Already Run ensures that it runs at most one time for a frame

  if already_run == False:
    # Uses Human Pose Analysis for Keypoints
    [fencer_data_temp, keypoints_temp] = human_pose_analysis(image)
    display(f'The fencer data is:')
    display(fencer_data)
    fencer_data.append(fencer_data_temp)
    keypoints.append(keypoints_temp)
    already_run = True

  return (fencer_data, keypoints, already_run)