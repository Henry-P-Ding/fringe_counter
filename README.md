# fringe_counter
 Counts elapsed fringes in Michelson inteferometry.

# Installation
Run the `install.sh` file with Python 3 installed.

# Setup
Write video files into the `video_list.txt` in the following format (separated by spaces):

```
video_path start_frame end_frame sample_number
```
with a separate line for each video. `end_frame` can be set to `-1` if the last frame should be the last frame of the video.  

# Running
Run the `run.sh` file.
