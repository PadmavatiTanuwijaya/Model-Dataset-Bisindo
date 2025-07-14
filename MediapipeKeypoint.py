import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2  # FIX: import langsung

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# Ukuran output tetap (kotak)
OUTPUT_SIZE = 300
EXTRA_PADDING = 40

# Output video writer
out = cv2.VideoWriter('output_segmented_landmarks.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      20,
                      (OUTPUT_SIZE, OUTPUT_SIZE))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # Buat canvas hitam ukuran output
    black_canvas = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=np.uint8)

    if results.multi_hand_landmarks:
        all_x, all_y = [], []

        # Ambil semua titik dari 1–2 tangan
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                all_x.append(int(lm.x * w))
                all_y.append(int(lm.y * h))

        if all_x and all_y:
            # Tentukan bounding box
            x_min = max(min(all_x) - EXTRA_PADDING, 0)
            x_max = min(max(all_x) + EXTRA_PADDING, w)
            y_min = max(min(all_y) - EXTRA_PADDING, 0)
            y_max = min(max(all_y) + EXTRA_PADDING, h)

            # Buat kotak square
            box_w = x_max - x_min
            box_h = y_max - y_min
            diff = abs(box_w - box_h)
            if box_w > box_h:
                y_min = max(y_min - diff // 2, 0)
                y_max = min(y_max + diff - diff // 2, h)
            else:
                x_min = max(x_min - diff // 2, 0)
                x_max = min(x_max + diff - diff // 2, w)

            crop_w = x_max - x_min
            crop_h = y_max - y_min

            # Skala ulang dan gambar ulang landmark ke canvas
            for hand_landmarks in results.multi_hand_landmarks:
                scaled_landmarks = []
                for lm in hand_landmarks.landmark:
                    x_px = int((lm.x * w - x_min) * OUTPUT_SIZE / crop_w)
                    y_px = int((lm.y * h - y_min) * OUTPUT_SIZE / crop_h)
                    scaled_landmarks.append(landmark_pb2.NormalizedLandmark(
                        x=x_px / OUTPUT_SIZE, y=y_px / OUTPUT_SIZE, z=lm.z
                    ))

                # Bungkus jadi NormalizedLandmarkList
                landmark_list = landmark_pb2.NormalizedLandmarkList(
                    landmark=scaled_landmarks
                )

                # Gambar ke canvas hitam
                mp_draw.draw_landmarks(
                    black_canvas,
                    landmark_list,
                    mp_hands.HAND_CONNECTIONS,
                    mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2)
                )

    # Simpan dan tampilkan
    out.write(black_canvas)
    cv2.imshow("Tangan di Tengah - Background Hitam", black_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
