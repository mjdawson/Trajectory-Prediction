import pandas as pd
import numpy as np
import csv


def generate_augmented_trajs(traj):
    # creates 5 new augmented trajectories for the given trajectory input
    o1 = [(-x,y) for (x,y) in traj]
    o2 = [(x,-y) for (x,y) in traj]
    o3 = [(-y,x) for (x,y) in traj]
    o4 = [(-x,-y) for (x,y) in traj]
    o5 = [(y,-x) for (x,y) in traj]
    o6 = [(y,x) for (x,y) in traj]
    o7 = [(-y,-x) for (x,y) in traj]
    return [o1, o2, o3, o4, o5, o6, o7]

def generate_orientations():
    # whether to flip about y-axis, number of 90-degree ccw rotations
    return [(False, 1),
            (False, 2),
            (False, 3),
            (True, 0),
            (True, 1),
            (True, 2),
            (True, 3)]

def load_simple_array(fname,augment=False):
    # loads the file into a simple array of frame index vs [x,y]
    trajectories = process_text_file_simple(fname)
    trajectory_list = []
    for key,val in trajectories.items():        
        trajectory_list.append(np.array(val))
        if (augment):        
            new_trajectories = generate_augmented_trajs(val)
            for t in new_trajectories:
                trajectory_list.append(np.array(t))
    return trajectory_list

def load_others_array(fname,N,centered=False):
    # loads the file into an array of frame index vs [x,y,flattened other array]    
    trajectories = process_text_file_others(fname)
    trajectory_list = []
    for piD,traj in trajectories.items():
        data = []
        for val in traj:
            my_x = val[0]
            my_y = val[1]
            others_list = val[2]
            others_array = np.zeros((N,N))
            for (x,y) in others_list:
                if centered:
                    Dx = x-my_x
                    Dy = y-my_yT                
                    ind_x = int(N/2+N/2*Dx)
                    ind_y = int(N/2+N/2*Dy)
                    if ind_x >= 0 and ind_x < N and ind_y >= 0 and ind_y < N:
                        others_array[ind_x,ind_y] = 1
                else:
                    ind_x = int(N*x)
                    ind_y = int(N*y)
                    if ind_x >= 0 and ind_x < N and ind_y >= 0 and ind_y < N:
                        others_array[ind_x,ind_y] = 1                    

            xy_array = np.array((my_x,my_y))
            #ret_array = np.concatenate((xy_array,others_array.flatten()),axis=0)
            #ret_array = np.reshape(ret_array,(2+N*N,1))
            data.append((xy_array, others_array))
        trajectory_list.append(data)
        #trajectory_list.append(np.concatenate(data,axis=1).T)
        #trajectory_list.append([(my_x,my_y),others_array])
    return trajectory_list

def load_full_array(fname,N):
    trajectory_list_with_others = load_others_array(fname,N)
    segment_array = get_segmented_image(fname, N=N)

    trajectory_list = []
    for t in trajectory_list_with_others:
        new_t = []
        for xy_array, others_array in t:
            me_array = get_my_array(xy_array, N)
            new_t.append((xy_array, me_array, others_array, segment_array))
        trajectory_list.append(new_t)
    #trajectory_list = [t.append(segment_array) for t in trajectory_list_with_others]
    #trajectory_list = [t.append(get_my_array(t, N)) for t in trajectory_list]
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
    df = pd.read_csv(fname + '.txt',delimiter=' ',names=['frame','personID','x-pos','y-pos'])
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

def get_my_array(trajectory, N):
    me_array = np.zeros((N,N))
    x = trajectory[0]
    y = trajectory[1]
    #(x,y), _ = trajectory
    x = float(x)
    y = float(y)    
    x = (x+1.0)/2       # normalize between 0 and 1
    y = (y+1.0)/2       # normalize between 0 and 1  
    nx = int(N*x)  # get array indeces
    ny = int(N*y)
    if nx >= 0 and nx < N and ny >= 0 and ny < N:
        me_array[nx,ny] = 1
    return me_array


def get_segmented_image(fname, N=10):
    with open(fname + '.csv', 'rU') as f:  #opens PW file
        reader = csv.reader(f)
        data = list(list(rec) for rec in csv.reader(f, delimiter=',')) #reads csv into a list of lists
        f.close() #close the csv
    segment_array = np.zeros((N,N))
    for x,y,seg_class in data:
        x = float(x)
        y = float(y)
        seg_class = int(seg_class)
        x = (x+1.0)/2       # normalize between 0 and 1
        y = (y+1.0)/2       # normalize between 0 and 1
        nx = int(N*x)       # get array indeces
        ny = int(N*y)
        if nx >= 0 and nx < N and ny >= 0 and ny < N:
            segment_array[nx,ny] = seg_class
    return segment_array

def load_full_augmented_data(fname, N):
    augmented_trajectories = []
  
    trajectories = load_full_array(fname, N)
    orientations = generate_orientations()

    augmented_trajectories += trajectories

    for traj in trajectories:
        for flip, num_rotations in orientations:
            new_traj = []
          
            for t in traj:
                coords, me, others, seg = t
                x, y = coords[0], coords[1]

                new_x, new_y = x, y
                new_me = me
                new_others = others
                new_seg = seg

                if flip:
                    new_x, new_y = -new_x, new_y
                    new_me = np.flip(new_me, axis=0)
                    new_others = np.flip(new_others, axis=0)
                    new_seg = np.flip(new_seg, axis=0)

                for _ in range(num_rotations):
                    tmp = new_x
                    new_x = -new_y
                    new_y = tmp
                    new_me = np.rot90(new_me)
                    new_others = np.rot90(new_others)
                    new_seg = np.rot90(new_seg)

                new_traj.append((np.array((new_x, new_y)), new_me, new_others, new_seg))

            augmented_trajectories.append(new_traj)

    return augmented_trajectories

        


if __name__ == '__main__':
    N = 10
    fname = 'train/nexus_3'
    traj = load_full_augmented_data(fname,N)
    print('=================== (x,y) ===================')
    print(traj[0][0])
    print('')
    print('=========== others occupancy grid ===========')
    print(traj[0][1])
    print('')    
    print('=========== segmented image file ============')
    print(traj[0][2])
    print('')    
    print('============= my occupancy grid =============')
    print(traj[0][3])
    print('')




