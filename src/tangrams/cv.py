#!/usr/bin/env python3
"""
apriltag_canny.py

Usage examples:
  # webcam (index 0)
  python apriltag_canny.py --source 0 --tag-size 0.05

  # video file
  python apriltag_canny.py --source demo.mp4 --tag-size 0.05

  # single image
  python apriltag_canny.py --source some_image.jpg --tag-size 0.05

  # with camera intrinsics for pose estimation (fx fy cx cy)
  python apriltag_canny.py --source 0 --tag-size 0.05 --fx 600 --fy 600 --cx 320 --cy 240
"""

import cv2
from pupil_apriltags import Detector

def main(
    SOURCE = 0, # camera device
    TAG_SIZE = 0.035, # in meters
    CAMERA_PARAMS = (992.00309549, 992.87050637, 620.77175804, 369.88717748), # fx, fy, cx, cy
    CAMERA_CAPTURE_PX = (1280, 720)
):
    detector = Detector(
        families='tag36h11',
        nthreads=4,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=True,
        decode_sharpening=0.25,
        debug=0
    )

    cap = cv2.VideoCapture(SOURCE)
    if cap is None or not cap.isOpened():
        print(f"Failed to open source: {SOURCE}")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Using DirectShow backend can help on Windows

    # Set your target resolution (e.g., 1920x1080)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CAPTURE_PX[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CAPTURE_PX[1])

    # Verify what the camera actually accepted
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Resolution set to: {width}x{height}")

    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        darker = frame
        #darker = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        gray = cv2.cvtColor(darker, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(gray,
            estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE)

        for detection in detections:
            if detection.tag_id != 0: continue
            print(f"Translation (X, Y, Z): {detection.pose_t[:, 0]}")

        # read current Canny / area parameters
        cv2.imshow("AprilTag + Tangram Detector", gray)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()