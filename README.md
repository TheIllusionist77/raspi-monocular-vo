# monocular_vo
A monocular, camera-based, visual odometry system designed to run on a Raspberry Pi using the WIDE Raspberry Pi Camera v3.

Note: If you're using any other camera, the intrinsic matrix will need to be re-calculated.

visual_odometry.py - Main visual odometry system based on Nicolai Nielsen's implementation.</br>
image_capture.py - Uses Picamera2 to take and save images on Raspberry Pi.</br>
camera_calibration.py - Program from the OpenCV documentation for calculating intrinsic matrices.</br>
intrinsic.npy - Numpy array storing the WIDE Raspberry Pi Camera v3's intrinsic matrix.</br>

The images folder contains the 16 checkerboard images used for calibration, which can be found at https://boofcv.org/notwiki/calibration/A4_chessboard.pdf.

calibration_result.jpg is the result of image0.jpg being adjusted for distortion using the intrinsic matrix.
