from ultralytics import YOLO
import cv2
import supervision as sv

video_file = "example-1.mp4"

#Enter the points between which the line needs to be drawn
start = sv.Point(0, 349)
end = sv.Point(638,140)

cap = cv2.VideoCapture(video_file)
# Get the video's frame dimensions and frame rate
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_rate = int(cap.get(5))

# Define the output video writer
output_video_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (frame_width, frame_height), isColor=True)

def main():
    #draws the line btw points
    line_counter = sv.LineZone(start=start, end=end)
    
    #labels the line with 'in' and 'out'
    line_annotator = sv.LineZoneAnnotator(thickness=1, text_thickness=1, text_scale=0.5)
    
    #defining the bounding box annotator
    box_annotator = sv.BoxAnnotator(
        thickness = 2,
        text_thickness =  1,
        text_scale = 0.5
    )
    #getting the yolov8s model
    model = YOLO("yolov8s.pt")
    
    #source = 0 for webcam
    for result in model.track(source = video_file, show = False, stream = True, agnostic_nms=True):
        frame = result.orig_img

        #detections contaains xyxy coordinates of bounding box, mask, confidence, class id and track id
        detections = sv.Detections.from_ultralytics(result)

        #getting detections only for person class (person class is 0 in coco dataset)
        detections = detections[detections.class_id == 0] 
     
        #if a person is detected, tracker id is stored in detections.tracker_id
        if result.boxes.id is not None:
            detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

        labels = [f"{tracker_id} {confidence: 0.2f}"
                   for _, mask, confidence, class_id, tracker_id
                   in detections]

        #drawing bounding box and labelling them 
        frame = box_annotator.annotate(scene = frame, detections = detections, labels = labels)

        #increases the count when detection box coordinates crosses the line
        line_counter.trigger(detections)
        line_annotator.annotate(frame = frame, line_counter = line_counter)
        cv2.imshow("yolov8", frame)
        out.write(frame)

        if (cv2.waitKey(30) == 27):
            break
        
    cap.release()
    out.release()

if __name__ == "__main__":
    main()

