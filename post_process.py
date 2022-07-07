import pandas as pd
from transform import transformer as tf
import os
import numpy as np
from scipy.interpolate import interp1d
import shutil


class Post_process:
    def __init__(self, pre_dir):
        self.pre_dir = pre_dir
        root, dirs, files = next(os.walk(self.pre_dir))
        self.demos = [os.path.join(root, dir) for dir in dirs]

        self.processed = []

        self.post_dir = './postprocessed/' + self.pre_dir.split('/')[-1]
        if not os.path.isdir(self.post_dir):
            os.makedirs(self.post_dir, exist_ok = False)

        return

    def transform(self, ndi_file):
        '''
        This function will convert the marker's pose to gripper's pose based on which marker is visible.

        Parameters
        ----------
        ndi_file: string
            The path to the ndi file that needs to be processed.

        Returns
        -------
        df: DataFrame
            A DataFrame that contains the gripper's pose trajectory.
        '''
        my_transformer = tf(ndi_file)
        transformed_data = my_transformer.process_file()
        df = pd.DataFrame.from_dict(transformed_data)
        return df

    def interpolate(self, transformed_data, kind = 'linear', max_gap = 0):
        '''
        This function will interpolate the gripper's pose trajectory.

        Parameters
        ----------
        transformed_data: DataFrame
            The DataFrame that contains the gripper's pose trajectory.
        kind: string
            The interpolate method. Options are: 'linear', 'nearest', 'nearest-up',‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
        max_gap : int, optional
            Maximum gap size to fill. By default, all gaps are interpolated.

        Returns
        -------
        interpolated_data: DataFrame
            A DataFrame that contains the interpolated_data gripper's pose trajectory.


        '''
        interpolated_data = transformed_data.copy().astype('float')
        xy = transformed_data.astype('float').values
        xy_filled = self._columnwise_interp(xy, kind, max_gap)
        filled = ~np.isnan(xy_filled)
        interpolated_data[filled] = xy_filled
        return interpolated_data

    def process_ndi(self):
        '''This method will transform the marker position to gripper pose and then do interpolation'''
        for demo in self.demos:
            root, dirs, files = next(os.walk(demo))
            ndi_files = []
            for f in files:
                if 'NDI_preprocessed.txt' in f:
                    ndi_files.append(os.path.join(root, f))
            if len(ndi_files) > 1:
                raise ('There are more than one ndi files for this demonstration.')
            else:
                ndi_file = ndi_files[0]
                df = self.transform(ndi_file)
                interpolated_data = self.interpolate(df)
                self.processed.append((ndi_file, interpolated_data))
        return

    def process_servo(self):
        '''It is simply copy the preprocessed servo file for now'''
        self.actions = {}
        for demo in self.demos:
            self.actions[demo] = {}
            root, dirs, files = next(os.walk(demo))
            for file in files:
                if 'Servo-displacement' in file:
                    servo_file = file
            with open(os.path.join(root, servo_file), 'r') as f:
                lines = f.readlines()
            action_start = []
            action_end = []
            for line in lines:
                if 'closed' in line:
                    action_start.append(line.split(',')[0])
                elif 'open' in line:
                    action_end.append(line.split(',')[0])
            self.actions[demo]['start'] = action_start
            self.actions[demo]['end'] = action_end
            src = root + '/' + servo_file
            dst = src.replace('preprocessed', 'postprocessed')
            shutil.copyfile(src, dst)

    def process_video(self):
        '''It is simply copy the preprocessed video file for now, might do something later'''
        for demo in self.demos:
            root, dirs, files = next(os.walk(demo))
            for d in dirs:
                if d == 'left' or d == 'right':
                    dir_full = os.path.join(root, d)
                    root_d, dirs_d, files_d = next(os.walk(dir_full))
                    src = os.path.join(root_d, files_d[0])
                    dst = src.replace('preprocessed', 'postprocessed')
                    dst_dir = os.path.dirname(dst)
                    if not os.path.isdir(dst_dir):
                        os.makedirs(dst_dir)
                    shutil.copyfile(src, dst)



    def _columnwise_interp(self, data, kind, max_gap=0):
        """
        Perform cubic spline interpolation over the columns of *data*.
        All gaps of size lower than or equal to *max_gap* are filled,
        and data slightly smoothed.

        Parameters
        ----------
        data : array_like
            2D matrix of data.
        kind: string
            The interpolate method. Options are: 'linear', 'nearest', 'nearest-up',‘zero’, ‘slinear’,
            ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’
            refer to a spline interpolation of zeroth, first, second or third order; ‘previous’ and ‘next’
            simply return the previous or next value of the point; ‘nearest-up’ and ‘nearest’ differ when interpolating
            half-integers (e.g. 0.5, 1.5) in that ‘nearest-up’ rounds up and ‘nearest’ rounds down. Default is ‘linear’.
        max_gap : int, optional
            Maximum gap size to fill. By default, all gaps are interpolated.

        Returns
        -------
        interpolated data with same shape as *data*
        """
        if np.ndim(data) < 2:
            data = np.expand_dims(data, axis=1)
        nrows, ncols = data.shape
        temp = data.copy()
        valid = ~np.isnan(temp)

        x = np.arange(nrows)
        for i in range(ncols):
            mask = valid[:, i]
            if (
                    np.sum(mask) > 3
            ):  # Make sure there are enough points to fit the cubic spline
                spl = interp1d(x[mask], temp[mask, i], kind=filtertype, fill_value='extrapolate')
                y = spl(x)
                if max_gap > 0:
                    inds = np.flatnonzero(np.r_[True, np.diff(mask), True])
                    count = np.diff(inds)
                    inds = inds[:-1]
                    to_fill = np.ones_like(mask)
                    for ind, n, is_nan in zip(inds, count, ~mask[inds]):
                        if is_nan and n > max_gap:
                            to_fill[ind: ind + n] = False
                    y[~to_fill] = np.nan
                temp[:, i] = y
        return temp

    def save_processed_file(self):
        '''This function will save the processed NDI file'''
        for item in self.processed:
            fname = item[0]
            processed_data = item[1]
            outname = fname.replace('preprocessed', 'postprocessed')
            outdir = os.path.dirname(outname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            processed_data.to_csv(outname)
        return



if __name__ == '__main__':
    pre_dir = './preprocessed/2022-05-26'
    pp = Post_process(pre_dir)
    # pp.process_servo()
    pp.process_video()
    # pp.process_ndi()
    # pp.save_processed_file()
