import os
from datetime import datetime
import pandas as pd
import sys
import pyzed.sl as sl
import numpy as np
import cv2
import enum
import os



class AppType(enum.Enum):
    LEFT_AND_RIGHT = 1
    LEFT_AND_DEPTH = 2
    LEFT_AND_DEPTH_16 = 3

def progress_bar(percent_done, bar_length=50):
    done_length = int(bar_length * percent_done / 100)
    bar = '=' * done_length + '-' * (bar_length - done_length)
    sys.stdout.write('[%s] %f%s\r' % (bar, percent_done, '%'))
    sys.stdout.flush()

def svo_to_mp4s(video, outdir, output_as_video = True):
    '''
    This function will convert the .svo file got from ZED to .mp4 file.

    parameters
    ----------
    video: string
        The path to the .svo file
    outdir: string
        The path to the directory where the .mp4 will be saved
    output_as_video: bool
        Not useful for now. Might be useful in the future.

    returns:
    -------
    0
    '''
    svo_input_path = video
    output_path = outdir
    app_type = AppType.LEFT_AND_RIGHT

    # Specify SVO path parameter
    init_params = sl.InitParameters()

    init_params.set_from_svo_file(str(svo_input_path))
    init_params.svo_real_time_mode = False  # Don't convert in realtime
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)

    # Create ZED objects
    zed = sl.Camera()

    # Open the SVO file specified as a parameter
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.stdout.write(repr(err))
        zed.close()
        exit()

    # Get image size
    image_size = zed.get_camera_information().camera_resolution
    width = image_size.width
    height = image_size.height

    # Prepare side by side image container equivalent to CV_8UC4
    svo_image_left_rgba = np.zeros((height, width, 4), dtype=np.uint8)
    svo_image_right_rgba = np.zeros((height, width, 4), dtype=np.uint8)

    # Prepare single image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    depth_image = sl.Mat()

    video_writer = None
    vid_id = os.path.basename(video).split('.')[0]
    outputdir_left = output_path + '/left'
    outputdir_right = output_path + '/right'
    os.makedirs(outputdir_left, exist_ok=True)
    os.makedirs(outputdir_right, exist_ok=True)
    output_path_left = outputdir_left + f'/{vid_id}-left.mp4'
    output_path_right = outputdir_right + f'/{vid_id}-right.mp4'
    if output_as_video:
        # Create video writer with MPEG-4 part 2 codec
        video_writer_left = cv2.VideoWriter((output_path_left),
                                            cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                            max(zed.get_camera_information().camera_fps, 25),
                                            (width, height))

        video_writer_right = cv2.VideoWriter((output_path_right),
                                             cv2.VideoWriter_fourcc('M', '4', 'S', '2'),
                                             max(zed.get_camera_information().camera_fps, 25),
                                             (width, height))

        if not video_writer_left.isOpened():
            sys.stdout.write("OpenCV video writer cannot be opened. Please check the .avi file path and write "
                             "permissions.\n")
            zed.close()
            exit()

    rt_param = sl.RuntimeParameters()
    rt_param.sensing_mode = sl.SENSING_MODE.FILL

    # Start SVO conversion to AVI/SEQUENCE
    sys.stdout.write("Converting SVO... Use Ctrl-C to interrupt conversion.\n")

    nb_frames = zed.get_svo_number_of_frames()
    while True:
        if zed.grab(rt_param) == sl.ERROR_CODE.SUCCESS:
            svo_position = zed.get_svo_position()

            # Retrieve SVO images
            zed.retrieve_image(left_image, sl.VIEW.LEFT)

            if app_type == AppType.LEFT_AND_RIGHT:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
            elif app_type == AppType.LEFT_AND_DEPTH:
                zed.retrieve_image(right_image, sl.VIEW.DEPTH)
            elif app_type == AppType.LEFT_AND_DEPTH_16:
                zed.retrieve_measure(depth_image, sl.MEASURE.DEPTH)

            if output_as_video:
                # Copy the left image to the left side of SBS image
                svo_image_left_rgba = left_image.get_data()

                # Copy the right image to the right side of SBS image
                svo_image_right_rgba = right_image.get_data()

                # Convert SVO image from RGBA to RGB
                ocv_image_left_rgb = cv2.cvtColor(svo_image_left_rgba, cv2.COLOR_RGBA2RGB)

                ocv_image_right_rgb = cv2.cvtColor(svo_image_right_rgba, cv2.COLOR_RGBA2RGB)

                # Write the RGB image in the video
                video_writer_left.write(ocv_image_left_rgb)
                video_writer_right.write(ocv_image_right_rgb)
            else:
                # Generate file names
                filename1 = output_path / ("left%s.png" % str(svo_position).zfill(6))
                filename2 = output_path / (("right%s.png" if app_type == AppType.LEFT_AND_RIGHT
                                            else "depth%s.png") % str(svo_position).zfill(6))

                # Save Left images
                cv2.imwrite(str(filename1), left_image.get_data())

                if app_type != AppType.LEFT_AND_DEPTH_16:
                    # Save right images
                    cv2.imwrite(str(filename2), right_image.get_data())
                else:
                    # Save depth images (convert to uint16)
                    cv2.imwrite(str(filename2), depth_image.get_data().astype(np.uint16))

        # Display progress
        progress_bar((svo_position + 1) / nb_frames * 100, 30)

        # Check if we have reached the end of the video
        if svo_position >= (nb_frames - 1):  # End of SVO
            sys.stdout.write("\nSVO end has been reached. Exiting now.\n")
            break
    print(f'The video id is {vid_id}')
    if output_as_video:
        # Close the video writer
        video_writer_left.release()
        video_writer_right.release()

    zed.close()
    return 0

class Pre_process:
    def __init__(self, raw_dir):
        self.raw_dir = raw_dir
        self.servo_files_list = []
        self.ndi_files_list = []
        self.videos_list = []
        self.suffix = '_preprocessed'
        root, dirs, files = next(os.walk(self.raw_dir))
        all_files = []
        for dir in dirs:
            all_files += [os.path.join(root, dir, file) for file in next(os.walk(os.path.join(root, dir)))[2]]
        for name in all_files:
            if "Servo-displacement" in name:
                self.servo_files_list.append(name)
            elif ".txt" in name:
                self.ndi_files_list.append(name)
            elif '.svo' in name:
                self.videos_list.append(name)
        self.matched_list = []
        self.match_servo_ndi_video_files()
        self.pre_dir = './preprocessed/' + self.raw_dir.split('/')[-1]
        if not os.path.isdir(self.pre_dir):
            os.mkdir(self.pre_dir )

    def match_servo_ndi_video_files(self):
        '''
        This function match the servo, ndi, and video files in self.raw_dir folder based on the prefix id.
        :return: A list tuples. Each entry is a matched instance which contains 4 elements: id, servo file, ndi file, and video file.
        '''
        for name in self.servo_files_list:
            servo_id = name.split("/")[-1].split('-')[0]
            matched_ndi_files = [ndi_file for ndi_file in self.ndi_files_list if servo_id in ndi_file]
            matched_video_files = [video_file for video_file in self.videos_list if servo_id in video_file]
            if len(matched_ndi_files) == 1 and len(matched_video_files) == 1:
                self.matched_list.append({'id': servo_id, 'servo': name, 'ndi': matched_ndi_files[0], 'video': matched_video_files[0]})


    def process_servo(self, save_to_csv = True):
        '''
        This function preprocess the servo data by:

        1. Convert the datetime to seconds for each timestamp.
        2. Extract only the timestamp where the gripper state changes.

        parameters
        ----------
        save_to_csv: bool
            Where or not the processed servo file will be saved as csv file

        returns
        -------
        df: DataFrame
            The DataFrame contains the processed servo file
        '''
        good_demo_ind = []
        for i, matched in enumerate(self.matched_list):
            servo_file = matched['servo']
            with open(servo_file, 'r') as f:
                lines = f.readlines()
            if not lines[-2][0] == 'S': # This is not a successful demonstration
                continue
            good_demo_ind.append(i)
            start = lines[-6].split(': ')[1].strip('\n')
            end = lines[-5].split(': ')[1].strip('\n')
            delta_t = self._compute_delta_t(start, end)
            print(f'The total duration of demonstration {matched["id"]} is: {delta_t} seconds.')
            matched['time_duration'] = delta_t
            # df = pd.DataFrame(columns = ['Timestamp', 'Gripper state'])
            dict = {'Timestamp': [], 'Gripper state': [] }
            for line in lines:
                if 'gripper open' in line or 'gripper closed' in line:
                    time_stamp = line.split(',')[0]
                    gripper_state = line.split(',')[1]
                    dt = self._compute_delta_t(start, time_stamp)
                    dict['Timestamp'].append(dt)
                    dict['Gripper state'].append(gripper_state.split()[1])
            matched_id = matched['id']
            outdir = self.pre_dir + f'/{matched_id}'
            if not os.path.isdir(outdir):
                os.mkdir(outdir)
            df = pd.DataFrame.from_dict(dict)
            fname = outdir + '/' + os.path.basename(servo_file) + self.suffix
            if save_to_csv:
                df.to_csv(fname, index = False)
        self.matched_list = [self.matched_list[i] for i in good_demo_ind]
        return df

    def process_video(self):
        '''
        This function covert the original .svo videos to 2 .mp4 videos(left camera and right camera)
        :return:
        '''
        for matched in self.matched_list:
            video_svo = matched['video']
            matched_id = matched['id']
            outdir = self.pre_dir + '/' + matched_id
            svo_to_mp4s(video_svo, outdir)

    def process_ndi(self):
        '''
        This function converts the datetime to seconds for each timestamp. It will also figure out which marker is visible to NDI.

        '''
        for matched in self.matched_list:
            matched_id = matched['id']
            ndi_file = matched['ndi']
            processed_lines = []
            with open( ndi_file, 'r') as f:
                lines = f.readlines()
            start = lines[6].split(',')[1]
            end = lines[-1].split(',')[1]
            time_total = self._compute_delta_t(start, end)
            frames_total = int(lines[-1].split(',')[0].split()[1])

            for line in lines[6:]:
                y = line.strip().split(",")
                frame_ind = int(y[0].split()[1])
                time_stamp = frame_ind / frames_total * time_total
                del y[0]
                y[0] = str(time_stamp)
                if "Both" not in y[1]:
                    if "449" in lines[0]:
                        if "339" in lines[1]:
                            y[1] = str(449)
                            y[9] = str(339)
                        else:
                            self.error = "Cannot pre-process NDI Labview file-Tool in line 2 should be 339"
                            return False

                    elif "339" in lines[0]:
                        if "449" in lines[1]:
                            y[9] = str(449)
                            y[1] = str(339)
                        else:
                            # raise IOError ("Cannot pre-process NDI Labview file-Tool in line 2 should be 449")
                            self.error = "Cannot pre-process NDI Labview file-Tool in line 2 should be 449"
                            return False
                    else:
                        # raise IOError ("Cannot pre-process NDI Labview file:Tool in line 1 should be 449 or 339")
                        self.error = "Cannot pre-process NDI Labview file:Tool in line 1 should be 449 or 339"
                        return False

                    if (float(y[2]) == 0.0) and (float(y[3]) == 0.0) and (
                            float(y[4]) == 0.0):  # just checking if x,y,z are zero which means no values
                        y[1] = y[9]
                        y[2] = y[10]
                        y[3] = y[11]
                        y[4] = y[12]
                        y[5] = y[13]
                        y[6] = y[14]
                        y[7] = y[15]
                        y[8] = y[16]

                    y = y[:9]
                    newline = ','.join(y)
                else:
                    newline = y[0] + ', NaN'
                processed_lines.append(newline)
            outdir = self.pre_dir + f'/{matched_id}'
            fname = outdir + '/' + os.path.basename(ndi_file).replace('.txt', f'_NDI{self.suffix}.txt')
            with open(fname, 'w') as f:
                for i in processed_lines:
                    i = i + '\n'
                    f.write(i)


    def _compute_delta_t(self, start, end):
        '''
        This function computes the time duration between start and end.

        parameters
        ----------
        start: string
            The starting time in format YYYY-MM-DD HH:MM:SS
        end: string
            The ending time in format YYYY-MM-DD HH:MM:SS
        returns
        -------
        delta_t: float
        the time difference between start and end in seconds.
        '''
        try:
            delta_t = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
        except ValueError as e:
            # Data from NDI is not in isoformat, instead it is in 'YYYY-MM-DD-HH-mm-ss'
            start = start.strip()
            end = end.strip()
            start_new = end_new = ''
            for i, c in enumerate(start):
                if i == 10:
                    start_new = start_new + ' '
                    end_new = end_new + ' '
                elif i == 13:
                    start_new = start_new + ':'
                    end_new = end_new + ':'
                elif i == 16:
                    start_new = start_new + ':'
                    end_new = end_new + ':'
                else:
                    start_new = start_new + c
                    end_new = end_new + end[i]
            delta_t = (datetime.fromisoformat(end_new) - datetime.fromisoformat(start_new)).total_seconds()
        return delta_t

    def process_servo_video_ndi(self):
        self.process_servo()
        self.process_ndi()
        self.process_video()




if __name__ == '__main__':

    raw_dir = './raw_data/Jun30-2022'
    pp = Pre_process(raw_dir)
    pp.process_servo_video_ndi()

#### Convert video from svo to mp4
    raw_dir = './convert_reference_frame/Jun28-2022/svo'
    raw_dir = './raw_data/Jun30-2022'
    root, dirs, files = next(os.walk(raw_dir))
    for f in files:
        fname = f.split('.')[0]
        video_path = os.path.join(root, f)
        outdir = os.path.join(os.path.dirname(video_path).replace('svo', 'mp4s'), fname)
        svo_to_mp4s(video_path, outdir)
