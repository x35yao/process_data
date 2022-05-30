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
        my_transformer = tf(ndi_file)
        transformed_data = my_transformer.process_file()
        df = pd.DataFrame.from_dict(transformed_data)
        return df

    def interpolate(self, transformed_data, kernel = 'linear', window_size = 0):
        # for item in self.transformed:
        interpolated_data = transformed_data.copy().astype('float')
        xy = transformed_data.astype('float').values
        xy_filled = self._columnwise_interp(xy, kernel, window_size)
        filled = ~np.isnan(xy_filled)
        interpolated_data[filled] = xy_filled
        return interpolated_data

    def batch_process(self):
        for demo in self.demos:
            root, dirs, files = next(os.walk(demo))
            ndi_files = []
            for f in files:
                if 'preprocessed.txt' in f:
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
        # It is simply copy the preprocessed servo file for now
        for demo in self.demos:
            root, dirs, files = next(os.walk(demo))
            for file in files:
                if 'Servo-displacement' in file:
                    servo_file = file
            src = root + '/' + servo_file
            dst = src.replace('preprocessed', 'postprocessed')
            shutil.copyfile(src, dst)

    def process_video(self):
        for demo in self.demos:
            root, dirs, files = next(os.walk(demo))
            vid_folder = os.path.join(root, dirs[0])
            root, dirs, files = next(os.walk(vid_folder))
            src = os.path.join(root, files[0])
            dst = src.replace('preprocessed', 'postprocessed')
            dst = dst.replace('/left/', '/')
            shutil.copyfile(src, dst)

    def _columnwise_interp(self, data, filtertype, max_gap=0):
        """
        Perform cubic spline interpolation over the columns of *data*.
        All gaps of size lower than or equal to *max_gap* are filled,
        and data slightly smoothed.

        Parameters
        ----------
        data : array_like
            2D matrix of data.
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
        for item in self.processed:
            fname = item[0]
            processed_data = item[1]
            outname = fname.replace('preprocessed', 'postprocessed').replace('.txt', '.csv')
            outdir = os.path.dirname(outname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            processed_data.to_csv(outname)
        return



if __name__ == '__main__':
    pre_dir = './preprocessed/2022-05-26'
    pp = Post_process(pre_dir)
    pp.batch_process()
    pp.save_processed_file()
    pp.process_servo()
    pp.process_video()