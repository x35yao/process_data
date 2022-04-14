from scipy.spatial.transform import Rotation as R
import numpy as np
import pandas as pd
#the gripper handle has two locations (top and bottom)
#Static transformation is to convert measurements of a rigid body from one tool to the other. These values are obtained by putting the 499 at the top and 339 close to the gripper center.
#Pose and location when 449 is in the top - v1,q1 are 449 and v2,q2 are 339 at the center. Used to convert 449 coordinates to gripper center corrdinates.
v1_449 = [97.663788, -180.389755, -1895.446655]
q1_449 = [0.416817, -0.806037, 0.028007, -0.419267]
v2_339 = [78.019791, -26.525036, -1980.021118]
q2_339 = [0.222542, 0.551251, 0.281243, 0.753326]

#pose and location when v1,q1 of 339 is at the bottom and v2,q2 of 449 at the center. Used to convert 339 coordinates to gripper center corrdinates.
v1_339 = [203.19, -58.99, -1621.9]
q1_339 = [0.7765, -0.2614, -0.5724, 0.032]
v2_449 = [107.71, -127.45, -1699.52]
q2_449 = [0.2803, -0.5491, 0.4564, -0.6415]

def homogenous_transform(R,vect):

    '''
    :param R: 3x3 matrix
    :param vect: list x,y,z
    :return:Homogenous transformation 4x4 matrix using R and vect
    '''

    H = np.zeros((4,4))
    H[0:3,0:3] = R
    frame_displacement = vect + [1]
    D = np.array(frame_displacement)
    D.shape = (1,4)
    H[:,3] = D
    return H

def inverse_homogenous_transform(H):

    '''
    :param H: Homogenous Transform Matrix
    :return: Inverse Homegenous Transform Matrix
    '''


    R = H[0:3,0:3]
    origin = H[:-1,3]
    origin.shape = (3,1)

    R = R.T
    origin = -R.dot(origin)
    return homogenous_transform(R,list(origin.flatten()))

def center_tool_339_to_gripper_center():

    '''
    The y-axis of 339 is aligned with the y axis of the gripper. The z-axis of the 339 will require a rotation of 90
    (counter clockwise with respect to y R (y,90) to get align gripper z axis to outward pointing. the origin of the
    339 needs to be moved in z-axis by + 40.45mm to get it to the origin of the gripper

    :return: homogenous transformation from 339 center to gripper center
    '''

    d =[0.0,0.0,40.45,1.0]
    H = np.zeros((4,4))
    H.shape = (4,4)
    H[:,3]= d
    H[(1,0),(1,2)]=1
    H[2,0]= -1
    return H

def center_tool_449_to_gripper_center():

    '''
    The y-axis of 4499 is aligned with the y axis of the gripper. The z-axis of the 449 will require a rotation of 90
    (counter clockwise with respect to y R (y,90) to get align gripper z axis to outward pointing. the origin of the
    449 needs to be moved in z-axis by + 35.36 (Not accurate) to get it to the origin of the gripper.
    the accurate measure from geometry 32.796

    :return: homogenous transformation from 339 center to gripper center
    '''

    d =[0.0,0.0,32.796,1.0]
    H = np.zeros((4,4))
    H.shape = (4,4)
    H[:,3]= d
    H[(1,0),(1,2)]=1
    H[2,0]= -1
    return H

def static_transform_449_top(q1,v1,q2,v2):
    '''

    :param q1: unit quaternions representing the rotation of the frame of 449 tool at the top
    :param v1: vector representing the rotation of the frame of 449 tool at the top
    :param q2: unit quaternions representing the rotation of the frame of 339 tool at the center
    :param v2: vector representing the rotation of the frame of 339 tool at the center
    :return: homogenous tranformation
    '''
    # H1 -  Transform a point in 449 frame to NDI frame
    # h1 -  Transform a point from NDI frame to 449 frame
    # H2 -  Transform a point from 339 frame to NDI frame
    # H3 -  Transform a point from TCP frame to 339 frame
    # H3 -  Homogenous transformation from the center tool frame to center of the gripper with axis rotated where the y
    # is parallel and between the two fingers and z is pointing outward


    R1 = R.from_quat(q1).as_matrix()
    H1 = homogenous_transform(R1, v1)
    h1 = inverse_homogenous_transform(H1)

    R2 = R.from_quat(q2).as_matrix()
    H2 = homogenous_transform(R2, v2)

    H3 = center_tool_339_to_gripper_center()
    H = (h1.dot(H2)).dot(H3)
    return H

def static_transform_339_bottom(q1,v1,q2,v2):
    '''

    :param q1: unit quaternions representing the rotation of the frame of 339 tool at the bottom of the gripper
    :param v1: vector representing the translation of the frame of 339 tool
    :param q2: unit quaternions representing the rotation of the frame of 449 tool at the center
    :param v2: vector representing the translation of the frame of 449
    :return: homogenous trsanformation
    '''
    # H1 -  Transform a point in 339 frame to NDI frame
    # h1 -  Transform a point from NDI frame to 339 frame
    # H2 -  Transform a point from 449 frame to NDI frame
    # H3 -  Transform a point from TCP frame to 449 frame
    # is parallel and between the two fingers and z is pointing outward

    R1 = R.from_quat(q1).as_matrix()
    H1 = homogenous_transform(R1, v1)
    h1 = inverse_homogenous_transform(H1)

    R2 = R.from_quat(q2).as_matrix()
    H2 = homogenous_transform(R2, v2)


    H3 = center_tool_449_to_gripper_center()
    H = (h1.dot(H2)).dot(H3)
    return H

class transformer:
    def __init__(self,labview_ndi_file):

        try:
            with open(labview_ndi_file) as f:
                self.lines = f.readlines()
        except:
            print("Unable to open file {}".format(labview_ndi_file))
            raise
            return
        self.fname = labview_ndi_file
        self.HT_from449_to_gripper_center = static_transform_449_top(q1_449, v1_449, q2_339, v2_339)
        self.HT_from339_to_gripper_center = static_transform_339_bottom(q1_339, v1_339, q2_449, v2_449)
        return

    def process_file(self):
        self.data = {'Time':[], 'x': [], 'y': [], 'z': [], 'Rx': [], 'Ry': [], 'Rz':[]}
        for line in self.lines:
            capture_time,tool = line.split(",")[0:2]
            if (tool == "449"):
                x, y, z, qr, qi, qj, qk = list(map(float,(line.strip().split(",")[2:])))
                R_tool = R.from_quat([qr, qi, qj, qk]).as_matrix()
                H_tool = homogenous_transform(R_tool, [x, y, z])
                H_gripper = H_tool.dot(self.HT_from449_to_gripper_center)            # pose and position of Gripper Center w.r.t NDI frame
                x_gripper = H_gripper[0,3]
                y_gripper = H_gripper[1,3]
                z_gripper = H_gripper[2,3]
                R_gripper = H_gripper[:3, :3]
                r = R.from_matrix(R_gripper).as_rotvec()
                Rx_gripper = r[0]
                Ry_gripper = r[1]
                Rz_gripper = r[2]
            elif (tool == "339"):
                x, y, z, qr, qi, qj, qk = list(map(float,(line.strip().split(",")[2:])))
                R_tool = R.from_quat([qr, qi, qj, qk]).as_matrix()
                H_tool = homogenous_transform(R_tool, [x, y, z])
                H_gripper = H_tool.dot(
                    self.HT_from339_to_gripper_center)  # pose and position of Gripper Center w.r.t NDI frame
                x_gripper = H_gripper[0, 3]
                y_gripper = H_gripper[1, 3]
                z_gripper = H_gripper[2, 3]
                R_gripper = H_gripper[:3, :3]
                r = R.from_matrix(R_gripper).as_rotvec()
                Rx_gripper = r[0]
                Ry_gripper = r[1]
                Rz_gripper = r[2]
            else:
                x_gripper = y_gripper = z_gripper = Rx_gripper = Ry_gripper = Rz_gripper = np.nan
            self.data['Time'].append(capture_time)
            self.data['x'].append(x_gripper)
            self.data['y'].append(y_gripper)
            self.data['z'].append(z_gripper)
            self.data['Rx'].append(Rx_gripper)
            self.data['Ry'].append(Ry_gripper)
            self.data['Rz'].append(Rz_gripper)
        return self.data

    def save_processed_file(self):
        df = pd.DataFrame.from_dict(self.data)
        df.to_csv(self.fname.replace('.txt','-transformed.csv'))

if __name__ == '__main__':
    labview_ndi_file = './preprocessed/260650/260650-2022-04-13-14-43-37_preprocessed.txt'
    my_transformer = transformer(labview_ndi_file)
    my_transformer.process_file()
    my_transformer.save_processed_file()