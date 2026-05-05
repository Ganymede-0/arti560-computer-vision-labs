import cv2
import time
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Run keypoint detection')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--video_file", default="sample_video.mp4", help="Input Video")

args = parser.parse_args()

MODE = "MPI"

if MODE == "COCO":
    protoFile = "./coco/pose_deploy_linevec.prototxt"
    weightsFile = "./coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE == "MPI" :
    protoFile = "./mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "./mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

inWidth = 368
inHeight = 368
threshold = 0.1

# ==========================================================
# BULLETPROOF VIDEO LOADING
# ==========================================================
current_folder = os.path.dirname(os.path.abspath(__file__))
input_source = os.path.join(current_folder, "..", "media", "walking-persons.mp4")

print(f"\n--- DEBUG: Looking for video at: {input_source} ---")

if not os.path.exists(input_source):
    print("🚨 FATAL ERROR: The video file does not exist at that path!")
    sys.exit()

cap = cv2.VideoCapture(input_source)

if not cap.isOpened():
    print("🚨 FATAL ERROR: OpenCV found the file, but cannot open it.")
    sys.exit()

hasFrame, frame = cap.read()
if not hasFrame:
    print("🚨 FATAL ERROR: Found the video, but couldn't read the first frame!")
    sys.exit()

print("✅ SUCCESS: Video loaded perfectly! Starting heavy CPU processing now (Please wait...)...\n")
# ==========================================================

save_name = os.path.splitext(os.path.basename(input_source))[0]
vid_writer = cv2.VideoWriter(f"{save_name}_openpose.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
if args.device == "cpu":
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

frame_count = 1

while cv2.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    
    # Check if video is over BEFORE trying to copy the frame
    if not hasFrame:
        print("\n✅ Finished processing all frames!")
        break
        
    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    points = []

    for i in range(nPoints):
        probMap = output[0, i, :, :]
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            points.append((int(x), int(y)))
        else :
            points.append(None)

    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    process_time = time.time() - t
    cv2.putText(frame, "time taken = {:.2f} sec".format(process_time), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    
    # Print a progress update so you know it hasn't frozen
    print(f"Processed frame {frame_count} in {process_time:.2f} seconds...")
    frame_count += 1

    vid_writer.write(frame)

cap.release()
vid_writer.release()