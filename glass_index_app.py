import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def get_bounds(crop_threshold, img):
    ip, jp = np.where(img > crop_threshold)
    return np.array([
        [np.min(ip), np.min(jp)],
        [np.max(ip), np.max(jp)]
    ])


def read_video(path, frame_range, sample_frames, crop_threshold=150, peak_threshold=15, box_fraction=(0.2, 0.1)):
    total_samples = len(sample_frames)
    vid = cv2.VideoCapture(path)
    if frame_range[1] > 0:
        end_frame = min(int(vid.get(cv2.CAP_PROP_FRAME_COUNT)), frame_range[1])
    else:
        end_frame = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    # setup figures
    if total_samples > 0:
        fig, axs = plt.subplots(total_samples, 2)
        fig.set_size_inches(6, 3 * total_samples)

    vid.set(cv2.CAP_PROP_POS_FRAMES, frame_range[0] - 1)
    frame_count = frame_range[0]
    sample_count = 0
    box_values = []
    while vid.isOpened() and frame_count <= end_frame:
        ret, frame = vid.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        frame = frame[:, :, 2] # get red channel
        if frame_count == frame_range[0]:
            bounds = get_bounds(crop_threshold, frame)
            cropped_frame = frame[bounds[0, 0]:bounds[1, 0], bounds[0, 1]:bounds[1, 1]]
            # perform sobley and smooth
            sobely = cv2.Sobel(cropped_frame, cv2.CV_64F, 2, 0) 
            sobley = cv2.blur(sobely, (10, 10))
            # get vertical slice position
            slice_y = int((bounds[1, 0] - bounds[0, 0]) / 2)
            # get slice peaks
            smoothed_slice = gaussian_filter1d(sobely[slice_y, :], sigma=15) 
            peaks = list(find_peaks(smoothed_slice)[0])
            box_bounds = np.array([
                [slice_y - box_fraction[1] * (bounds[1, 0] - bounds[0, 0]) / 2 + bounds[0, 0],  (1 - box_fraction[0]) / 2 * (peaks[-1] - peaks[-2]) + peaks[-2] + bounds[0, 1]],
                [slice_y + box_fraction[1] *(bounds[1, 0] - bounds[0, 0]) / 2 + bounds[0, 0], -(1 - box_fraction[0]) / 2 * (peaks[-1] - peaks[-2]) + peaks[-1] + bounds[0, 1]]
            ]).astype(int)
        
        # get frame frames in the box
        box_frame = frame[box_bounds[0, 0]:box_bounds[1, 0], box_bounds[0, 1]:box_bounds[1, 1]]
        box_values.append(np.average(box_frame))

        # get samples
        if frame_count in sample_frames and sample_count < total_samples:
            # cropped image
            axs[sample_count, 0].set_xlim((bounds[0, 1], bounds[1, 1]))
            axs[sample_count, 0].set_ylim((bounds[0, 0], bounds[1, 0]))
            axs[sample_count, 0].imshow(frame, cmap="gray", aspect="auto")
            axs[sample_count, 0].plot([box_bounds[0, 1], box_bounds[0, 1], box_bounds[1, 1], box_bounds[1, 1]], [box_bounds[0, 0], box_bounds[1, 0], box_bounds[1, 0], box_bounds[0, 0]], 'rx')
            axs[sample_count, 0].set_title(f"Frame  {frame_count}")

            # box image
            axs[sample_count, 1].set_xlim((box_bounds[0, 1], box_bounds[1, 1]))
            axs[sample_count, 1].set_ylim((box_bounds[0, 0], box_bounds[1, 0]))
            axs[sample_count, 1].imshow(frame, cmap="gray", aspect="auto")
            axs[sample_count, 1].set_title(f"Frame  {frame_count}")
            sample_count += 1
        frame_count += 1
    
    if total_samples > 0:
        fig.tight_layout()
        fig.savefig(f"{path}_samples.png")
    return np.array(box_values)


video_paths = []
frame_ranges = []
sample_totals = []
with open("video_list.txt") as f:
    for line in f:
        path, start_frame, end_frame, total = line.split(" ")
        video_paths.append(path)
        frame_ranges.append((int(start_frame), int(end_frame)))
        sample_totals.append(int(total))


for i, path in enumerate(video_paths):
    print(f"Processing {path}")
    frame_range = frame_ranges[i]
    total_frames = sample_totals[i]
    if total_frames == 0:
        sample_frames = []
    else:
        sample_frames = [frame_range[0] + int((frame_range[1] - frame_range[0]) / total_frames * i) for i in range(0, total_frames + 1)]
    box_values = -1 * read_video(path, frame_range, sample_frames=sample_frames, crop_threshold=100) # flip sign

    fig, ax = plt.gcf(), plt.gca()
    fig = plt.figure(figsize=(20, 4))
    smoothed_box_values = gaussian_filter1d(box_values, sigma=10)
    plt.plot(smoothed_box_values)
    peaks = find_peaks(smoothed_box_values, prominence=50, height=0.6 * np.max(smoothed_box_values) + 0.4 * np.min(smoothed_box_values))[0]
    plt.plot(peaks, smoothed_box_values[peaks], 'ro')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Avg Box Pixel Value")
    fig.suptitle(f"{path} Box Pixel Sum, {peaks.shape[0]} Peaks")
    fig.savefig(f"{path}_peaks.png")