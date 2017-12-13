import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(trajectory1, trajectory2):
  x1 = list(trajectory1[:,0])
  y1 = list(trajectory1[:,1])
  x2 = list(trajectory2[:,0])
  y2 = list(trajectory2[:,1])

  plt.figure()
  axes = plt.gca()
  axes.set_xlim([-1.0,1.0])
  axes.set_ylim([-1.0,1.0])
  plt.plot(x1, y1, 'bo-')
  plt.plot(x2, y2, 'ro-')
  plt.show()

if __name__ == '__main__':
  traj1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.6, 0.4], [0.7, 0.7]])
  traj2 = np.array([[0.0, 0.3], [0.3, 0.5], [0.7, 0.3], [0.8, 0.9]])
  traj3 = np.array([[0.15, 0.2], [0.5, 0.5], [0.5, 0.4], [0.8, 0.7]])
  traj4 = np.array([[0.2, 0.1], [0.0, 0.1], [0.5, 0.6], [0.9, 0.9]])
  plot_trajectories(traj1, traj2)
  plot_trajectories(traj3, traj4)

