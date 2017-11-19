import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import csv

def load_simple_array(fname):
    # loads the file into a simple array of frame index vs [x,y]
    trajectories = process_text_file_simple(fname)
    trajectory_list = []
    for key,val in trajectories.iteritems():        
        trajectory_list.append(np.array(val))
    return trajectory_list

def load_others_array(fname,N):
    # loads the file into an array of frame index vs [x,y,flattened other array]    
    trajectories = process_text_file_others(fname)
    trajectory_list = []
    for piD,traj in trajectories.iteritems():
        data = []
        for val in traj:
            my_x = val[0]
            my_y = val[1]
            others_list = val[2]
            others_array = np.zeros((N,N))
            for (x,y) in others_list:
                Dx = x-my_x
                Dy = y-my_y
                ind_x = int(N/2+Dx)
                ind_y = int(N/2+Dy)
                if ind_x >= 0 and ind_x < N and ind_y >= 0 and ind_y < N:
                    others_array[ind_x,ind_y] = 1
            xy_array = np.array((my_x,my_y))
            ret_array = np.concatenate((xy_array,others_array.flatten()),axis=0)
            ret_array = np.reshape(ret_array,(2+N*N,1))
            data.append(ret_array)
        trajectory_list.append(np.concatenate(data,axis=1).T)
    return trajectory_list

def get_max_x_y_positions(df):
    # gets the maximum absolute value of the x and y positions in the dataset (for normalization)    
    x_max_abs = 0.0
    y_max_abs = 0.0
    for row in df.iterrows():
        _, _, x_pos, y_pos = row[1]
        if abs(x_pos) > x_max_abs:
            x_max_abs = x_pos
        if abs(y_pos) > y_max_abs:
            y_max_abs = y_pos
    return x_max_abs, y_max_abs

def process_text_file_simple(fname):
    # Load CSV into pandas dataframe (and normalize positions)
    df = pd.read_csv(fname,delimiter=' ',names=['frame','personID','x-pos','y-pos'])
    df.head()
    num_rows = df.shape[0]
    x_max_abs, y_max_abs = get_max_x_y_positions(df)
    df['x-pos'] = df['x-pos'].apply(lambda x: x/x_max_abs)
    df['y-pos'] = df['y-pos'].apply(lambda y: y/y_max_abs)

    # Load trajectories into simple dictionary { personID -> [(frame1, x1, y1), (frame2, x2, y2), ...]}
    trajectories_simple = {}
    for row in df.iterrows():
        frame, pID, x_pos, y_pos = row[1]
        if pID not in trajectories_simple.keys():
            trajectories_simple[pID] = []
        trajectories_simple[pID].append((x_pos, y_pos))
    return trajectories_simple

def process_text_file_others(fname):
    # Load CSV into pandas dataframe (and normalize positions)
    df = pd.read_csv(fname,delimiter=' ',names=['frame','personID','x-pos','y-pos'])
    df.head()
    num_rows = df.shape[0]
    x_max_abs, y_max_abs = get_max_x_y_positions(df)
    df['x-pos'] = df['x-pos'].apply(lambda x: x/x_max_abs)
    df['y-pos'] = df['y-pos'].apply(lambda y: y/y_max_abs)

    # Load trajectories into dictionary with other people { personID -> [(frame1, x1, y1, other_ppl_array), (frame2, x2, y2, other_ppl_array), ...]}
    trajectories_others = {}
    for row in df.iterrows():
        frame, pID, x_pos, y_pos = row[1]
        if pID not in trajectories_others.keys():
            trajectories_others[pID] = []

        other_rows_in_frame = df.loc[df['frame'] == frame]
        other_people_data = other_rows_in_frame[['personID', 'x-pos', 'y-pos']]
        other_list = [(person[1],person[2]) for person in other_people_data.values if person[0] != pID]
        trajectories_others[pID].append((x_pos, y_pos,other_list))

    return trajectories_others
