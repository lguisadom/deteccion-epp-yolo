"""
Inferencia del verificador de cumplimiento de EPP sobre un video (extensión de
06_predict_images.py a secuencias). Recorre el video fotograma a fotograma con OpenCV,
aplica el mismo modelo y módulo compliance_checker, y produce un video anotado, fotogramas
de muestra y un CSV con la tasa de cumplimiento por fotograma.

Uso (desde la raíz del repositorio):
    python src/09_predict_video.py                       # data/videos/demo.mp4 (por defecto)
    python src/09_predict_video.py data/videos/obra.mp4  # ruta explícita
    python src/09_predict_video.py --webcam              # cámara 0

Salida: cada fuente se guarda en su propia subcarpeta para no sobrescribir corridas previas:
    outputs/compliance/video/<nombre>/
        <nombre>_anotado.mp4        video completo anotado
        frames/frame_XXXXXX.jpg     fotogramas de muestra
        compliance_por_frame.csv    tasa de cumplimiento por fotograma
(La webcam usa la subcarpeta "webcam".)
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

# Permite importar compliance_checker.py (está en la misma carpeta src/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compliance_checker import PPEComplianceChecker, Detection, annotate_frame  # noqa: E402

# --------------------------------------------------------------------------------
# CONFIGURACIÓN (edítala aquí si lo necesitas)
# --------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    # Modelo entrenado (el mismo que usan los scripts 05 y 06).
    "weights": PROJECT_ROOT / "outputs" / "training" / "yolov8s_baseline" / "weights" / "best.pt",
    # Video de entrada por defecto (puedes sobreescribirlo por línea de comandos).
    "video": PROJECT_ROOT / "data" / "videos" / "demo.mp4",
    # Carpeta donde se guardan todas las salidas.
    "out_dir": PROJECT_ROOT / "outputs" / "compliance" / "video",
    # Umbral de confianza mínimo para quedarse con una detección.
    "conf": 0.25,
    # GPU (0) o CPU ("cpu").
    "device": 0,
    # Cada cuántos fotogramas se guarda uno como evidencia en frames/.
    # (No guardamos todos para no llenar el disco; 1 de cada 15 es razonable.)
    "save_every": 15,
}


def detections_from_result(result, names) -> list:
    """Convierte el resultado de YOLO de UN fotograma en una lista de Detection.

    Idéntico a 06_predict_images.py: por cada caja toma su clase, confianza y las
    coordenadas xyxy en píxeles.
    """
    dets = []
    if result.boxes is None:
        return dets
    for box in result.boxes:
        cls_id = int(box.cls)
        dets.append(Detection(
            cls_name=names[cls_id],
            conf=float(box.conf),
            xyxy=tuple(float(v) for v in box.xyxy[0].tolist()),
        ))
    return dets


def main() -> None:
    if not CONFIG["weights"].exists():
        raise SystemExit(
            f"No existe el modelo: {CONFIG['weights']}\n"
            "Entrena primero: python src/04_train.py"
        )

    # 1) Resolver la fuente del video (archivo o webcam) desde la línea de comandos.
    use_webcam = "--webcam" in sys.argv
    extra_args = [a for a in sys.argv[1:] if not a.startswith("--")]
    video_path = Path(extra_args[0]) if extra_args else CONFIG["video"]
    source = 0 if use_webcam else str(video_path)

    if not use_webcam and not video_path.exists():
        print(f"[ERROR] No encuentro el video: {video_path}")
        print("        Copia tu archivo en data/videos/ o pasa la ruta como argumento:")
        print("        ../.venv/bin/python src/09_predict_video.py data/videos/tu_video.mp4")
        sys.exit(1)

    # 2) Preparar carpetas de salida.
    #    Cada fuente tiene su PROPIA subcarpeta, para no sobrescribir resultados de otros
    #    videos: outputs/compliance/video/<nombre_del_video>/. Para la webcam usamos un
    #    nombre fijo ("webcam"). Así puedes correr video1, video2, video3... y conservar
    #    la evidencia de cada uno por separado.
    run_name = "webcam" if use_webcam else video_path.stem
    out_dir = CONFIG["out_dir"] / run_name
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Salida de esta corrida: {out_dir}")

    # 3) Cargar el modelo y el verificador (misma configuración que 06).
    print(f"[INFO] Cargando modelo: {CONFIG['weights']}")
    model = YOLO(str(CONFIG["weights"]))
    names = model.names
    checker = PPEComplianceChecker(min_conf=CONFIG["conf"])

    # 4) Abrir el video con OpenCV.
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] No pude abrir la fuente de video: {source}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not use_webcam else -1
    print(f"[INFO] Video: {width}x{height} @ {fps:.1f} fps, {total} fotogramas")

    # 5) Preparar el escritor del video anotado (mp4). Lleva el nombre de la fuente para
    #    identificarlo fácilmente al descargarlo (ej. video1_anotado.mp4).
    out_video_path = out_dir / f"{run_name}_anotado.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

    # 6) Preparar el CSV de evidencia (tasa de cumplimiento por fotograma).
    csv_path = out_dir / "compliance_por_frame.csv"
    csv_file = open(csv_path, "w", encoding="utf-8")
    csv_file.write("frame,personas,cumplen,tasa_cumplimiento\n")

    # 7) Bucle principal: leer -> detectar -> verificar -> anotar -> escribir.
    frame_idx = 0
    total_personas = 0
    total_cumplen = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break  # se acabó el video

            result = model.predict(frame, conf=CONFIG["conf"],
                                   device=CONFIG["device"], verbose=False)[0]
            dets = detections_from_result(result, names)

            # check_frame devuelve (lista de PersonStatus, tasa[0..1] o None si no hay personas)
            statuses, rate = checker.check_frame(dets)
            n_personas = len(statuses)
            n_ok = sum(1 for st in statuses if st.compliant)
            total_personas += n_personas
            total_cumplen += n_ok
            tasa_txt = f"{rate:.3f}" if rate is not None else ""
            csv_file.write(f"{frame_idx},{n_personas},{n_ok},{tasa_txt}\n")

            # annotate_frame dibuja cajas verdes/rojas + banner y devuelve el fotograma.
            annotated = annotate_frame(frame, statuses, rate)
            writer.write(annotated)

            # Guardar un fotograma de muestra como evidencia ligera.
            if frame_idx % CONFIG["save_every"] == 0:
                cv2.imwrite(str(frames_dir / f"frame_{frame_idx:06d}.jpg"), annotated)

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[INFO] Procesados {frame_idx} fotogramas...")
    finally:
        cap.release()
        writer.release()
        csv_file.close()

    # 8) Resumen final.
    print("\n========== RESUMEN ==========")
    print(f"Fotogramas procesados : {frame_idx}")
    if total_personas:
        tasa_global = 100 * total_cumplen / total_personas
        print(f"Personas (acumulado)  : {total_personas}")
        print(f"Cumplen (acumulado)   : {total_cumplen}")
        print(f"Tasa de cumplimiento  : {tasa_global:.1f}%")
    print(f"Video anotado         : {out_video_path}")
    print(f"Fotogramas evidencia  : {frames_dir}/")
    print(f"CSV por fotograma     : {csv_path}")


if __name__ == "__main__":
    main()
