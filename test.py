import cv2
import numpy as np

# Apply a sepia filter to an image
def apply_sepia(image):
    sepia_filter = np.array([[0.272, 0.534, 0.131],
                             [0.349, 0.686, 0.168],
                             [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255).astype(np.uint8)
    return sepia_image

# Apply a kaleidoscopic effect
def apply_doctor_strange(frame):
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2

    # Create a rotation matrix for a kaleidoscope effect
    rotation_matrix = cv2.getRotationMatrix2D((center_x, center_y), angle=45, scale=1.0)
    rotated_frame = cv2.warpAffine(frame, rotation_matrix, (w, h))

    # Split into quadrants and mirror them
    top_left = rotated_frame[0:center_y, 0:center_x]
    bottom_left = cv2.flip(top_left, 0)
    top_right = cv2.flip(top_left, 1)
    bottom_right = cv2.flip(top_left, -1)

    # Combine quadrants
    frame[0:center_y, 0:center_x] = top_left
    frame[center_y:h, 0:center_x] = bottom_left
    frame[0:center_y, center_x:w] = top_right
    frame[center_y:h, center_x:w] = bottom_right

    return frame

# Apply a thermal vision effect
def apply_thermal_vision(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thermal_frame = cv2.applyColorMap(gray_frame, cv2.COLORMAP_JET)
    return thermal_frame

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    print("Press the following keys to switch modes:")
    print("'n': Normal Mode")
    print("'g': Grayscale Mode")
    print("'e': Edge Detection Mode")
    print("'s': Sepia Mode")
    print("'b': Blurred Mode")
    print("'d': Doctor Strange Mode")
    print("'t': Thermal Vision Mode")
    print("'q': Quit the application.")

    filter_mode = None  # Variable to store the selected filter mode

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Apply selected filter
        if filter_mode == 'grayscale':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif filter_mode == 'edges':
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(gray_frame, 50, 150)
        elif filter_mode == 'sepia':
            frame = apply_sepia(frame)
        elif filter_mode == 'blurred':
            frame = cv2.GaussianBlur(frame, (15, 15), 0)
        elif filter_mode == 'doctor_strange':
            frame = apply_doctor_strange(frame)
        elif filter_mode == 'thermal':
            frame = apply_thermal_vision(frame)

        # Display on-screen instructions
        overlay = frame.copy()
        cv2.putText(overlay, "Press 'q' to Quit | 'g': Grayscale | 'e': Edges | 's': Sepia | 'b': Blur | 'd': Strange | 't': Thermal | 'n': Normal",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Merge overlay with the frame
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        # Display the resulting frame
        cv2.imshow('Enhanced Camera', frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Quit
            break
        elif key == ord('g'):  # Grayscale
            filter_mode = 'grayscale'
            print("Switched to Grayscale Mode.")
        elif key == ord('e'):  # Edge Detection
            filter_mode = 'edges'
            print("Switched to Edge Detection Mode.")
        elif key == ord('s'):  # Sepia
            filter_mode = 'sepia'
            print("Switched to Sepia Mode.")
        elif key == ord('b'):  # Blurred
            filter_mode = 'blurred'
            print("Switched to Blurred Mode.")
        elif key == ord('d'):  # Doctor Strange
            filter_mode = 'doctor_strange'
            print("Switched to Doctor Strange Mode.")
        elif key == ord('t'):  # Thermal Vision
            filter_mode = 'thermal'
            print("Switched to Thermal Vision Mode.")
        elif key == ord('n'):  # Normal
            filter_mode = None
            print("Switched to Normal Mode.")

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
