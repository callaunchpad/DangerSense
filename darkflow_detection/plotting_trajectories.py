object_trajectories = # load from file
object_paths = []
for key in object_trajectories:
	path = object_trajectories[key]
	x_points = [obj['x'] for obj in path]
	y_points = [obj['y'] for obj in path]
	plt.scatter(x_points, y_points)
	plt.show()