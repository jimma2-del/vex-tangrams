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
    SOURCE = 0,
    
    TAG_SIZE = 0.05,
    
    FX = 600,
    FY = 600,
    CX = 320,
    CY = 240,
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

    cam_params = None
    if FX and FY and CX and CY:
        cam_params = (FX, FY, CX, CY)

    print("Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #darker = cv2.convertScaleAbs(frame, alpha=0.3, beta=0)
        darker = frame
        gray = cv2.cvtColor(darker, cv2.COLOR_BGR2GRAY)

        if cam_params:
            detections = detector.detect(gray,
                estimate_tag_pose=True, camera_params=cam_params, tag_size=TAG_SIZE)
        else:
            detections = detector.detect(gray)

        for detection in detections:
            print(f"Translation (X, Y, Z): {detection.pose_t}")

        # read current Canny / area parameters
        cv2.imshow("AprilTag + Tangram Detector", gray)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()