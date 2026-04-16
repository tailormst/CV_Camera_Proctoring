import cv2
import time
import math as m
import mediapipe as mp

# ================= UTILITY FUNCTIONS ================= #

def findDistance(x1, y1, x2, y2):
    return m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def findAngle(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1

    if dy == 0:
        return 0

    angle = m.degrees(m.atan2(abs(dx), abs(dy)))
    return angle


# ALERT CONTROL (avoid spam)
last_warning_time = 0
WARNING_COOLDOWN = 5  # seconds

def sendWarning():
    global last_warning_time
    current_time = time.time()

    if current_time - last_warning_time > WARNING_COOLDOWN:
        print("⚠️ WARNING: Bad posture detected for too long!")
        last_warning_time = current_time


# ================= INITIALIZATION ================= #

good_frames = 0
bad_frames = 0

font = cv2.FONT_HERSHEY_SIMPLEX

# Colors
green = (127, 255, 0)
red = (50, 50, 255)
yellow = (0, 255, 255)
pink = (255, 0, 255)
light_green = (127, 233, 100)

# Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)


# ================= MAIN ================= #

if __name__ == "__main__":

    file_name = "input.mp4"  # use 0 for webcam
    cap = cv2.VideoCapture(file_name)

    if not cap.isOpened():
        print("Switching to webcam...")
        cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    print("Processing... Press 'q' to quit")

    while cap.isOpened():

        success, image = cap.read()
        if not success:
            print("End of video")
            break

        h, w = image.shape[:2]

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # ===== SAFETY CHECK =====
        if results.pose_landmarks is None:
            cv2.imshow("Posture Detection", image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
            continue

        lm = results.pose_landmarks.landmark
        lmPose = mp_pose.PoseLandmark

        # ================= LANDMARKS ================= #

        l_shldr = lm[lmPose.LEFT_SHOULDER]
        r_shldr = lm[lmPose.RIGHT_SHOULDER]
        l_ear = lm[lmPose.LEFT_EAR]
        l_hip = lm[lmPose.LEFT_HIP]

        l_shldr_x, l_shldr_y = int(l_shldr.x * w), int(l_shldr.y * h)
        r_shldr_x, r_shldr_y = int(r_shldr.x * w), int(r_shldr.y * h)
        l_ear_x, l_ear_y = int(l_ear.x * w), int(l_ear.y * h)
        l_hip_x, l_hip_y = int(l_hip.x * w), int(l_hip.y * h)

        # ================= ALIGNMENT ================= #

        offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

        if offset < 100:
            cv2.putText(image, "Aligned", (w - 150, 30), font, 0.8, green, 2)
        else:
            cv2.putText(image, "Not Aligned", (w - 150, 30), font, 0.8, red, 2)

        # ================= ANGLES ================= #

        neck = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        angle_text = f"Neck: {int(neck)}  Torso: {int(torso)}"

        # ================= POSTURE ================= #

        if neck < 40 and torso < 10:
            good_frames += 1
            bad_frames = 0
            color = light_green
        else:
            bad_frames += 1
            good_frames = 0
            color = red

        cv2.putText(image, angle_text, (10, 30), font, 0.8, color, 2)

        # ================= TIME ================= #

        good_time = good_frames / fps
        bad_time = bad_frames / fps

        if good_time > 0:
            cv2.putText(image, f"Good: {round(good_time,1)}s",
                        (10, h - 20), font, 0.8, green, 2)
        else:
            cv2.putText(image, f"Bad: {round(bad_time,1)}s",
                        (10, h - 20), font, 0.8, red, 2)

        # ================= ALERT ================= #

        if bad_time > 10:   # change to 180 later
            sendWarning()

        # ================= DRAW ================= #

        cv2.circle(image, (l_shldr_x, l_shldr_y), 6, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 6, yellow, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 6, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 6, pink, -1)

        cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), color, 3)
        cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), color, 3)

        # ================= SAVE ================= #

        video_output.write(image)

        # ================= DISPLAY ================= #

        cv2.imshow("Posture Detection", image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # ================= CLEANUP ================= #

    cap.release()
    video_output.release()
    cv2.destroyAllWindows()

    print("Finished Successfully!")