from django.shortcuts import render
import cv2
import asyncio
from django.http import StreamingHttpResponse
from ultralytics import YOLO
import cv2
import supervision as sv
from .models import CountData
from datetime import datetime  # Import datetime for timestamp

def video_stream(request):
    # Replace 'rtsp://your_rtsp_url_here' with your actual RTSP URL
    rtsp_url = 'people_counting_after_5.mp4'
    cap = cv2.VideoCapture(rtsp_url)

    #Enter the points between which the line needs to be drawn
    # start = sv.Point(1238,140)
    # end = sv.Point(0, 349)
    start = sv.Point(638,140)
    end = sv.Point(0, 349)

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
    count_data = CountData()
    if not cap.isOpened():
        return render(request, 'error.html')

    def process_frame(frame):
        if line_counter is None:
            initialize(frame)

        processed_frame=frame.copy()

        for result in model.track(frame,persist=True,agnostic_nms=True):
            detections = sv.Detections.from_ultralytics(result)
            detections = detections[detections.class_id == 0]

            #if a person is detected, tracker id is stored in detections.tracker_id
            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)

            labels = [f"{tracker_id} {confidence:0.2f}"
                    for _, mask, confidence, class_id, tracker_id
                    in detections]

            processed_frame = box_annotator.annotate(scene=processed_frame, detections=detections, labels=labels)

            p_in_count = line_counter.in_count
            p_out_count = line_counter.out_count
            line_counter.trigger(detections)
            # Update in_count and out_count in the database along with timestamp
            # if(line_counter.in_count>p_in_count):
            #     count_data.in_count = line_counter.in_count-p_in_count
            #     count_data.out_count = 0
            #     count_data.timestamp = datetime.now()
            #     count_data.save()
            # if(line_counter.out_count>p_out_count):
            #     count_data.in_count = 0
            #     count_data.out_count = line_counter.out_count-p_out_count
            #     count_data.timestamp = datetime.now()
            #     count_data.save()
                    
            line_annotator.annotate(frame=processed_frame, line_counter=line_counter)

        return processed_frame
    
    def generate():
        i=0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if(i%5==0):
                processed_frame = process_frame(frame)
                # Convert the frame to JPEG format
                _, jpeg = cv2.imencode('.jpg', processed_frame)
                frame_bytes = jpeg.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            i+=1
    result = StreamingHttpResponse(generate(), content_type="multipart/x-mixed-replace;boundary=frame")
    return result

def video_feed(request):
    response =  video_stream(request)
    return response

def index(request, *args, **kwargs):
    return render(request, 'index.html')

