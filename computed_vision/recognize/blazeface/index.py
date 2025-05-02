import cv2
import numpy as np
import onnxruntime as ort
import os
import mediapipe as mp

# Função para preprocessar a imagem para o modelo
def preprocess(face_img):
    img = cv2.resize(face_img, (128, 128))
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # Normaliza para [-1, 1]
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Função para calcular similaridade cosseno
def cosine_similarity(a, b):
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot / (norm_a * norm_b)

# Carrega modelo ONNX
session = ort.InferenceSession('./models/recognition_resnet27.onnx', providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Inicializar MediaPipe BlazeFace
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)

# Banco de dados de embeddings conhecidos (simulado)
known_faces = {}

# Função para registrar uma pessoa
def register_face(label, face_img):
    img_input = preprocess(face_img)
    embedding = session.run([output_name], {input_name: img_input})[0].flatten()
    known_faces[label] = embedding

# Função para detectar rostos usando MediaPipe BlazeFace
def detect_faces_blazeface(image):
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    detections = []
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            xmin = int(bboxC.xmin * w)
            ymin = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            xmax = xmin + width
            ymax = ymin + height
            detections.append((ymin, xmin, ymax, xmax))
    return detections

# Função para carregar imagens e gerar embeddings
def embeddingFaces():
    faces_dir = './faces'
    for person in os.listdir(faces_dir):
        person_path = os.path.join(faces_dir, person)
        if os.path.isfile(person_path) and person.endswith('.png'):
            face_img = cv2.imread(person_path)
            if face_img is not None:
                detections = detect_faces_blazeface(face_img)

                if len(detections) == 0:
                    print(f"Nenhum rosto detectado para {person}.")
                    continue

                for ymin, xmin, ymax, xmax in detections:
                    cropped_face = face_img[ymin:ymax, xmin:xmax]
                    label = os.path.splitext(person)[0]  # Nome da pessoa sem extensão
                    print(f"Registrando {label}...")
                    register_face(label, cropped_face)
                    print(f"Embedding gerado para {label}.")

embeddingFaces()

# Inicializar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_faces_blazeface(frame)

    for ymin, xmin, ymax, xmax in detections:
        face_img = frame[ymin:ymax, xmin:xmax]

        img_input = preprocess(face_img)
        embedding = session.run([output_name], {input_name: img_input})[0].flatten()

        # Comparar com rostos conhecidos
        name = "Desconhecido"
        best_score = 0.0
        for label, known_emb in known_faces.items():
            score = cosine_similarity(embedding, known_emb)
            if score > best_score and score > 0.5:
                best_score = score
                name = label

        # Desenhar retângulo e nome
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(frame, name, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Reconhecimento Facial', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
