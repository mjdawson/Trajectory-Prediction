from numpy import sqrt


def compute_linear_error(trajectories, Nf):
    dataset_error = 0.0
    traj_errors = {}

    for pID, traj in trajectories.iteritems():
        num_frames = len(traj)
        if num_frames <= Nf:
            continue
        else:
            t_prev = traj[Nf-1][0]
            x_prev = traj[Nf-1][1]
            y_prev = traj[Nf-1][2]        
            delta_t = traj[Nf-1][0] - traj[Nf-2][0]
            delta_x = traj[Nf-1][1] - traj[Nf-2][1]
            delta_y = traj[Nf-1][2] - traj[Nf-2][2]
            vel_x = delta_x/delta_t
            vel_y = delta_y/delta_t
            
            traj_error = 0.0
            for j in range(Nf,num_frames):
                t = traj[j][0]
                x = traj[j][1]
                y = traj[j][2]
                
                x_pred = x_prev + vel_x*(t-t_prev)
                y_pred = y_prev + vel_y*(t-t_prev)
                
                # note, this is using the initial linear model as the predictor.
                x_prev = x_pred
                y_prev = y_pred
                t_prev = t
                
                # may also use new x information to predict
                x_prev = x
                y_prev = y
                t_prev = t   
                error = sqrt((x-x_pred)**2 + (y-y_pred)**2)
                traj_error += error
            traj_errors[pID] = traj_error/(num_frames-Nf)
            dataset_error += traj_error
    return dataset_error/len(trajectories)