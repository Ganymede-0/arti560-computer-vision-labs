import time
import torch
import cv2
import numpy as np
from torchvision import transforms
import os
import sys

from utils.datasets import letterbox
from utils.general  import non_max_suppression_kpt
from utils.plots    import output_to_keypoint, plot_skeleton_kpts

def pose_video(frame):
    mapped_img = frame.copy()
    # Letterbox resizing.
    img = letterbox(frame, input_size, stride=64, auto=True)[0]
    img_ = img.copy()
    # Convert the array to 4D.
    img = transforms.ToTensor()(img)
    # Convert the array to Tensor.
    img = torch.tensor(np.array([img.numpy()]))
    # Load the image into the computation device.
    img = img.to(device)
    
    # Gradients are stored during training, not required while inference.
    with torch.no_grad():
        t1 = time.time()
        output, _ = model(img)
        t2 = time.time()
        fps = 1/(t2 - t1)
        output = non_max_suppression_kpt(output, 0.25, 0.65, nc=1, nkpt=17, kpt_label=True)
        output = output_to_keypoint(output)

    # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    nimg = img[0].permute(1, 2, 0) * 255
    nimg = nimg.cpu().numpy().astype(np.uint8)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)

    for idx in range(output.shape[0]):
        plot_skeleton_kpts(nimg, output[idx, 7:].T, 3)
        
    return nimg, fps

#------------------------------------------------------------------------------#
# Change forward pass input size.
input_size = 256

#---------------------------INITIALIZATIONS------------------------------------#

# Select the device based on hardware configs.
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print('Selected Device : ', device)

# Load keypoint detection model.
weights = torch.load('yolov7-w6-pose.pt', map_location=torch.device('cpu'), weights_only=False)
model = weights['model']
# Load the model in evaluation mode.
_ = model.float().eval()
# Load the model to computation device [cpu/gpu/tpu]
model.to(device)

save_name = "walking-persons"

# 1. Dynamically build the absolute, un-breakable path
current_folder = os.path.dirname(os.path.abspath(__file__))
vid_path = os.path.join(current_folder, "..", "media", f"{save_name}.mp4")

if not os.path.exists(vid_path):
    print("🚨 FATAL ERROR: The video file does not exist at that path!")
    sys.exit()

cap = cv2.VideoCapture(vid_path)

if not cap.isOpened():
    print("🚨 FATAL ERROR: OpenCV found the file, but cannot open it.")
    sys.exit()

# --- THE 6KB VIDEO FIX ---
# Read the first frame and process it to get the EXACT output dimensions
ret, frame = cap.read()
if not ret:
    print("🚨 FATAL ERROR: Found the video, but couldn't read the first frame!")
    sys.exit()

print("✅ SUCCESS: Video loaded perfectly! Calculating new dimensions...\n")

# Process the first frame to see what size YOLO makes it
first_processed_img, _ = pose_video(frame)
out_h, out_w, _ = first_processed_img.shape

# Initialize VideoWriter with the NEW dimensions!
out = cv2.VideoWriter(f"{save_name}_yolo7.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 10, (out_w, out_h))

# Write the first frame we already read
cv2.putText(first_processed_img, 'YOLOv7', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
out.write(first_processed_img[...,::-1])

if __name__ == '__main__':
    frame_count = 1
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print('\n✅ Finished processing all frames!')
            break

        img, fps_ = pose_video(frame)

        cv2.putText(img, 'FPS : {:.2f}'.format(fps_), (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)
        cv2.putText(img, 'YOLOv7', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)

        out.write(img[...,::-1])
        
        frame_count += 1
        print(f"Processed YOLO frame {frame_count}...", end="\r")

    cap.release()
    out.release()
    cv2.destroyAllWindows()