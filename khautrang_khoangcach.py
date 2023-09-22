
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from pyimagesearch import social_distancing_config as config
from pyimagesearch.detection import detect_people
from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2
import os
import time


from tkinter import *
from PIL import Image,ImageTk

tk=Tk()
tk.title("Phong Chong Covid-19")
# tk.geometry("670x400+0+0")
tk.geometry("670x500+0+0")
tk.resizable(0,0)
tk.configure(background="white")

lb01=Label(tk,fg="blue",bg="white",font="Times 17",text="Đề Tài: Nhận Dạng Vi Phạm Khẩu Trang Và Khoảng Cách")
lb01.pack()
lb01.place(x=60,y=10)

lb01=Label(tk,fg="black",bg="white",font="Times 12",text="Camera giám sát khẩu trang")
lb01.pack()
lb01.place(x=30,y=50)

lb01=Label(tk,fg="black",bg="white",font="Times 12",text="Camera giám sát khoảng cách")
lb01.pack()
lb01.place(x=360,y=50)

lb01=Label(tk,fg="black",bg="white",font="Times 13",text="Tổng số người:")
lb01.pack()
lb01.place(x=30,y=340)

lb01=Label(tk,fg="black",bg="white",font="Times 13",text="Số người vi phạm khoảng cách:")
lb01.pack()
lb01.place(x=30,y=370)

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence >0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
weightsPath = os.path.sep.join(["face_detector",
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
 
# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = load_model("mask_detector.model")
# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
# ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
# ln = np.array([ln[i[0] - 1] for i in net.getUnconnectedOutLayers()])

# initialize the video stream and pointer to output video file
print("Truy cap video...")
vs1 = cv2.VideoCapture("0.mp4")
# vs2 = VideoStream(src=0).start()  # dong nay de chạy webcam khẩu trang
vs2 = cv2.VideoCapture("mark.mp4")  # thay dong nay de doi sang video mong muon
time.sleep(2.0)
dem = 0
# loop over the frames from the video stream


while True:
    # frame2 = vs2.read()  # dong nay de chạy webcam khẩu trang
    ret, frame2 = vs2.read()     # thay dong nay de doi sang video mong muon
    frame2 = imutils.resize(frame2, width=600) 

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
    (locs, preds) = detect_and_predict_mask(frame2, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
    for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
        label = "Co Khau trang" if mask > withoutMask else "Khong khau trang"
        if label == "Co Khau trang":
            color = (0, 255, 0) 
        else:
            color =(0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
        cv2.putText(frame2, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame2, (startX, startY), (endX, endY), color, 2)

	# show the output frame
    cv2.imwrite('frame2.jpg',frame2)
    imagelg=Image.open('frame2.jpg')
    imagelg=imagelg.resize((270,260),Image.ANTIALIAS)
    imagelg=ImageTk.PhotoImage(imagelg)
    lb=Label(image=imagelg)
    lb.image=imagelg
    lb.pack()
    lb.place(x=30,y=70)
    tk.update()
    dem = dem + 1
	# read the next frame from the file
    ret, frame = vs1.read()
    if dem == 10:
        dem = 0
	    # resize the frame and then detect people (and only people) in it
        frame = imutils.resize(frame, width=600)
        results = detect_people(frame, net, ln,personIdx=LABELS.index("person"))

        violate = set()
        if len(results) >= 2:
		    # extract all centroids from the results and compute the
		    # Euclidean distances between all pairs of the centroids
            centroids = np.array([r[2] for r in results])
            #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
            D = dist.cdist(centroids, centroids, metric="euclidean")
		    # loop over the upper triangular of the distance matrix
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
				    # check to see if the distance between any two
				    # centroid pairs is less than the configured numberq
				    # of pixels
                    if D[i, j] < 40:
					    # update our violation set with the indexes of
					    # the centroid pairs
                        violate.add(i)
                        violate.add(j)

    	# loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
		    # extract the bounding box and centroid coordinates, then
		    # initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            color = (0, 255, 0) 
    		# if the index pair exists within the violation set, then
    		# update the color
            if i in violate:
                color = (0, 0, 255)

    		# draw (1) a bounding box around the person and (2) the
    		# centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    	# draw the total number of social distancing violations on the
	    # output frame
        lb01=Label(tk,fg="green",bg="white",font="Times 16",text=len(results))
        lb01.pack()
        # lb01.place(x=150,y=298)
        lb01.place(x=150,y=338)

        lb01=Label(tk,fg="red",bg="white",font="Times 16",text=len(violate))
        lb01.pack()
        # lb01.place(x=250,y=328)
        lb01.place(x=250,y=368)
	# check to see if the output frame should be displayed to our
	# screen
	
	# show the output frame
        cv2.imwrite('frame.jpg',frame)
        imagelg=Image.open('frame.jpg')
        imagelg=imagelg.resize((270,260),Image.ANTIALIAS)
        imagelg=ImageTk.PhotoImage(imagelg)
        lb=Label(image=imagelg)
        lb.image=imagelg
        lb.pack()
        lb.place(x=360,y=70)
        tk.update()
    
vs1.release()
vs2.release()
cv2.destroyAllWindows()