def youtube_clip(link):
  # Uses a given youtube_link to download and store the clip to the Sync Folder
  # Returns clip title and data if the clip duration is too long to download the entire clip

  path = '/content/drive/My Drive/Sync/'
  os.chdir(path)

  # Gets the Link for the Youtube Clip
  youtube_clip = link.split('=')[0][0:-2]
  # Uses the YouTube Query to get the time in seconds of the position in the clip
  youtube_clip_position = int(link.split('=')[1])
  # Clip End Position
  youtube_end_position = int(youtube_clip_position + 10)
  
  ydl_opts = {}
  #Gets info for the YouTube clip
  with youtube_dl.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(link, download=False)
    video_title = info_dict.get('title', None)
    file_size = info_dict.get('formats',None)[0]['filesize']
    duration = info_dict.get('duration',None)
    # container = info_dict.get('formats',None)[0]['container']

  # Initializes the container for the video file from YouTube
  container = 0
  
  display(f'The video title is {video_title}.')
  path = f'./{video_title}.mp4'

  # ydl_opts.update({'outtmpl':path})
  ydl_opts.update({'outtmpl':path})

  if verbose == True:
    display(f'The clip duration is {duration}.')

  if container == 'm4a_dash':
    display(f'The youtube link is a streamed video, downloading entire stream.')

  video_title_clip = video_title + '_clip.mp4'

  if verbose == True:
    display(f'The video title is:')
    display(video_title)

  # Downloads the entire file if the video clip is short or a DASH file, i.e. streamed video clip
  if duration < duration_download_limit or container == 'm4a_dash':
    #Downloads and saves the YouTube clip if within the duration limit
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download([link])

    # The video download may be .mp4 or mkv
    if exists(video_title + '.mp4'):
      video_title = video_title + '.mp4'
    if exists(video_title + '.mkv'):
      video_title = video_title + '.mkv'

    # if verbose == True:
    display(f'Duration is less than {duration_download_limit}. Downloading entire clip.')
    display(f'The video_title is {video_title}.')

    # Renames the video without spaces to accomodate ffmpeg
    os.rename(video_title, 'source_video.mp4')

    # !ffmpeg -y -ss $youtube_clip_position -i source_video.mp4 -c:v libx264 -c:a aac -frames:v 100 output_video.mp4

    # Python Compatible
    source_code = '!ffmpeg -y -ss $youtube_clip_position -i source_video.mp4 -c:v libx264 -c:a aac -frames:v 100 output_video.mp4'
    get_ipython().run_line_magic("sx", source_code )

    os.rename('output_video.mp4', video_title_clip)
    os.remove('source_video.mp4')
    duration_limit_data = [False, 'None']
  else:
    display(f'Duration is greater than {duration_download_limit}. Downloading a portion of the clip.')
    !ffmpeg $(youtube-dl -g $link | sed "s/.*/-ss $youtube_clip_position -i &/") -y -to $youtube_end_position -c copy out.mp4
    os.rename('out.mp4', video_title_clip)

    # [starting_frame, fps, total_frames] = determine_starting_frame(video_title_clip, touch_folder, True)
    os.chdir('/content/')
    frames_per_second = 25
    # start_time = starting_frame/frames_per_second
    duration_limit_data = [True, frames_per_second]

  return (video_title_clip, duration_limit_data)