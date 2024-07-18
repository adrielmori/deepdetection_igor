import cv2

def verificar_caminho(file_path):
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return None, 'Não foi possível abrir o vídeo.'
    return cap, None
