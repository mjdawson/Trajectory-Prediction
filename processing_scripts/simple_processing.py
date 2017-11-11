import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import csv

def get_max_x_y_positions(df):
    x_max_abs = 0.0
    y_max_abs = 0.0
    for row in df.iterrows():
        _, _, x_pos, y_pos = row[1]
        if abs(x_pos) > x_max_abs:
            x_max_abs = x_pos
        if abs(y_pos) > y_max_abs:
            y_max_abs = y_pos
    return x_max_abs, y_max_abs


def process_text_file(fname):
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
        trajectories_simple[pID].append((frame, x_pos, y_pos))


    # Load trajectories into dictionary with other people { personID -> [(frame1, x1, y1, other_ppl_array), (frame2, x2, y2, other_ppl_array), ...]}
    trajectories_others = {}
    for row in df.iterrows():
        frame, pID, x_pos, y_pos = row[1]
        if pID not in trajectories_others.keys():
            trajectories_others[pID] = []

        other_rows_in_frame = df.loc[df['frame'] == frame]
        other_people_data = other_rows_in_frame[['personID', 'x-pos', 'y-pos']]
        other_list = [tuple(person) for person in other_people_data.values if person[0] != pID]
        trajectories_others[pID].append((frame, x_pos, y_pos,other_list))

    return trajectories_simple, trajectories_others
