from numpy import sqrt


def compute_linear_error(trajectories, Nf):
    # Computes total error with a linear model
    #   inputs:
    #     - trajectories: list of numpy arrays with indeces i,j for frame index and [x,y,others_list elements], respectively
    #     - Nf: number of frames to observe before computing error
    #   outputs:
    #     - error : sqrt of sum of square loss (normalized by trajectory length)
    dataset_error = 0.0
    num_trajs = trajectories.shape[0]
    for i in range(num_trajs):
        traj = trajectories[i, :, :]
        num_frames,num_people = traj.shape
        if num_frames <= Nf:
            raise ValueError("Nf needs to be shorter than the number of frames in the trajectory")
            return 0
        if Nf < 3:
            raise ValueError("Nf needs to be greater than 3 for prediction")
            return 0
        x_prev = traj[Nf-1][0]
        y_prev = traj[Nf-1][1]        
        delta_x = traj[Nf-1][0] - traj[Nf-2][0]
        delta_y = traj[Nf-1][1] - traj[Nf-2][1]
        
        traj_error = 0.0
        for j in range(Nf,num_frames):
            x = traj[j][0]
            y = traj[j][1]
            
            x_pred = x_prev + delta_x
            y_pred = y_prev + delta_y
            
            # note, this is using the initial linear model as the predictor.
            x_prev = x_pred
            y_prev = y_pred
            
            # may also use new x information to predict
            #x_prev = x
            #y_prev = y

            error = sqrt((x-x_pred)**2 + (y-y_pred)**2)
            traj_error += error

        dataset_error += traj_error
    return dataset_error/len(trajectories)


