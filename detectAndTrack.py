import cv2

from imutils.video import FPS


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create(),
	"kcf": cv2.TrackerKCF_create(),
	"boosting": cv2.TrackerBoosting_create(),
	"mil": cv2.TrackerMIL_create(),
	"tld": cv2.TrackerTLD_create(),
	"medianflow": cv2.TrackerMedianFlow_create(),
	"mosse": cv2.TrackerMOSSE_create()
}
	
multiTracker = None
cap = cv2.VideoCapture(0)
fps = None
frameCount = 0

while(cap.isOpened()):

	ret, frame = cap.read()
	frameCount = frameCount + 1
	
	
	(H, W) = frame.shape[:2]
	


	if ret == True:
		
		frame = cv2.flip(frame,1)
		
		if multiTracker is None or frameCount % 100 == 0:
			faces = face_cascade.detectMultiScale(frame, 1.1, 5, minSize=(10, 10))
			
			if len(faces) > 0:
				multiTracker = cv2.MultiTracker_create()
				for (x, y, w, h) in faces:
					cv2.putText(frame,'Detected',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2,cv2.LINE_AA)
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
					multiTracker.add(cv2.TrackerKCF_create(), frame, (x, y, w, h))
					fps = FPS().start()
	
		else:
			success, boxes = multiTracker.update(frame)
			
			if len(boxes) == 0:
				frameCount = 99
				
			if success:	
				for box in boxes:		
					(x, y, w, h) = [int(v) for v in box]
					cv2.putText(frame,'Tracked',(x,y), cv2.FONT_HERSHEY_SIMPLEX, 2,(255,0,0),2,cv2.LINE_AA)
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
			fps.update()
			fps.stop()
			info = [
			("Tracker", "KCF"),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
			]
			
			for (i, (k, v)) in enumerate(info):
				text = "{}: {}".format(k, v)
				cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
			
		cv2.imshow("Frame",frame)
		
		
		
			
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break			
		
		
	else:
		break

cap.release()
cv2.destroyAllWindows()
	






