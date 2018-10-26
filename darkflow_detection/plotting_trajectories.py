import pickle
import matplotlib.pyplot as plt
import cv2

with open('object_trajectories.pickle', 'rb') as handle:
	object_trajectories = pickle.load(handle)
with open('last_image.pickle', 'rb') as handle:
	image = pickle.load(handle)

cap = cv2.VideoCapture("../snippet.mp4")

cap.set(1, cap.get(7)-50)
#sets position to the last frame

ret, img = cap.read()
cap.release()

print(object_trajectories)
print(len(object_trajectories))
object_paths = []
for key in object_trajectories:
	path = object_trajectories[key]
	x_points = [obj['x'] for obj in path]
	y_points = [obj['y'] for obj in path]
	plt.scatter(x_points, y_points)
plt.imshow(img)
plt.show()
