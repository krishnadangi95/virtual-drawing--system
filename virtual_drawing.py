
import cv2
import mediapipe as mp
import numpy as np

# ---------------- INITIALIZATION ----------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not found.")
    raise SystemExit

canvas = None
previous_x, previous_y = None, None

colors = {
    "G": (0, 255, 0),
    "R": (0, 0, 255),
    "B": (255, 0, 0),
    "W": (255, 255, 255)
}

current_color = colors["G"]
brush_size = 5
eraser_mode = False

# ---------------- MAIN PROGRAM ----------------

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            print("Unable to read webcam.")
            break

        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape

        if canvas is None:
            canvas = np.zeros_like(frame)

        # Detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        drawing_mode = False

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            lm = hand.landmark

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_up = lm[16].y < lm[14].y
            pinky_up = lm[20].y < lm[18].y

            # Draw when only index finger is raised
            drawing_mode = (
                index_up and not middle_up
                and not ring_up and not pinky_up
            )

            x = int(lm[8].x * width)
            y = int(lm[8].y * height)

            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)

            if drawing_mode:
                if previous_x is not None:
                    color = (
                        (0, 0, 0)
                        if eraser_mode else current_color
                    )
                    thickness = 30 if eraser_mode else brush_size

                    cv2.line(
                        canvas,
                        (previous_x, previous_y),
                        (x, y),
                        color,
                        thickness
                    )

                previous_x, previous_y = x, y
            else:
                previous_x, previous_y = None, None
        else:
            previous_x, previous_y = None, None

        # Combine webcam and canvas
        output = cv2.add(frame, canvas)

        # ---------------- USER INTERFACE ----------------

        # Top title bar
        cv2.rectangle(output, (0, 0), (width, 45), (45, 35, 25), -1)

        cv2.putText(
            output,
            "VIRTUAL DRAWING SYSTEM",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Current status
        if eraser_mode:
            status = "ERASER MODE"
        elif drawing_mode:
            status = "DRAWING"
        else:
            status = "READY - RAISE INDEX FINGER"

        cv2.putText(
            output,
            status,
            (15, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        # Toolbar instructions
        cv2.rectangle(
            output,
            (0, height - 100),
            (width, height),
            (35, 35, 35),
            -1
        )

        cv2.putText(
            output,
            "COLORS: G-Green  R-Red  B-Blue  W-White",
            (10, height - 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            "BRUSH: 1-3  |  E-Eraser  |  S-Save",
            (10, height - 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.putText(
            output,
            "C-Clear Canvas  |  Q-Quit",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1
        )

        cv2.imshow("Virtual Drawing System", output)

        # ---------------- KEYBOARD CONTROLS ----------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("g"):
            current_color = colors["G"]
            eraser_mode = False

        elif key == ord("r"):
            current_color = colors["R"]
            eraser_mode = False

        elif key == ord("b"):
            current_color = colors["B"]
            eraser_mode = False

        elif key == ord("w"):
            current_color = colors["W"]
            eraser_mode = False

        elif key == ord("1"):
            brush_size = 3

        elif key == ord("2"):
            brush_size = 7

        elif key == ord("3"):
            brush_size = 12

        elif key == ord("e"):
            eraser_mode = True

        elif key == ord("s"):
            filename = "my_drawing.png"
            saved = cv2.imwrite(filename, canvas)

            if saved:
                print(f"Drawing saved as {filename}")
            else:
                print("Unable to save drawing.")

        elif key == ord("c"):
            canvas = np.zeros_like(frame)
            previous_x, previous_y = None, None

        elif key == ord("q"):
            break

# ---------------- RELEASE RESOURCES ----------------

cap.release()
cv2.destroyAllWindows()
print("Virtual Drawing System closed.")
