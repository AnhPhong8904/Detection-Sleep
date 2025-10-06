import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    # Vẽ landmark cho từng khuôn mặt
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        # Convert landmark thành protobuf
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in face_landmarks
        ])

        # Vẽ các kết nối khuôn mặt
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    return annotated_image


def plot_face_blendshapes_bar_graph(face_blendshapes):
    # Tách tên và điểm số
    names = [category.category_name for category in face_blendshapes]
    scores = [category.score for category in face_blendshapes]
    ranks = range(len(names))

    fig, ax = plt.subplots(figsize=(12, 12))
    bar = ax.barh(ranks, scores, label=[str(x) for x in ranks])
    ax.set_yticks(ranks, names)
    ax.invert_yaxis()

    # Ghi điểm số trên từng thanh
    for score, patch in zip(scores, bar.patches):
        plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

    ax.set_xlabel("Score")
    ax.set_title("Face Blendshapes")
    plt.tight_layout()
    plt.show()


# ========================
# MAIN PROGRAM
# ========================

# Đọc ảnh
img = cv2.imread(r"video\1.jpg")
cv2.imshow("Input", img)

# Load model Mediapipe FaceLandmarker
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Detect landmark từ ảnh
mp_image = mp.Image.create_from_file(r"video\1.jpg")
detection_result = detector.detect(mp_image)
if not detection_result.face_landmarks:
    print("Không phát hiện khuôn mặt")
else:
    print("Phát hiện khuôn mặt")

# Vẽ landmark lên ảnh
annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)

# Hiển thị kết quả
cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

# Vẽ biểu đồ blendshapes
plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

# In transformation matrix
print("Facial Transformation Matrix:")
print(detection_result.facial_transformation_matrixes)
