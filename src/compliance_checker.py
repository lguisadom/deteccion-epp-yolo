"""
Verificador de cumplimiento de EPP basado en reglas (APORTE PROPIO del proyecto).

YOLO detecta objetos sueltos (Person, Hardhat, Safety Vest, ...). Este módulo aplica una
lógica geométrica para decidir, POR CADA PERSONA, si cumple con el EPP, y calcula la tasa
de cumplimiento de cada fotograma.

Diseño:
  - Es independiente del modelo: opera sobre detecciones genéricas (clase, confianza, caja).
    Así se puede testear sin YOLO (ver self-test en __main__) y reutilizar en imágenes/video.

Regla de cumplimiento:
  Una persona CUMPLE si tiene casco (Hardhat) en su región de cabeza Y chaleco (Safety Vest)
  en su torso, y no presenta violaciones explícitas (NO-Hardhat / NO-Safety Vest).
  Criterio conservador (pro-seguridad): si falta evidencia de un EPP, se marca NO conforme.

Limitaciones (para el informe): la asociación es geométrica y depende de la calidad del
detector y de los umbrales de región; no maneja perspectiva 3D ni oclusión total.
"""

from dataclasses import dataclass, field

# Nombres de clase relevantes del dataset Construction Site Safety
CLS_PERSON = "Person"
CLS_HARDHAT = "Hardhat"
CLS_NO_HARDHAT = "NO-Hardhat"
CLS_VEST = "Safety Vest"
CLS_NO_VEST = "NO-Safety Vest"


@dataclass
class Detection:
    """Una detección genérica de YOLO (en píxeles, formato xyxy)."""
    cls_name: str
    conf: float
    xyxy: tuple  # (x1, y1, x2, y2)


@dataclass
class PersonStatus:
    """Resultado del análisis de cumplimiento de una persona."""
    box: tuple
    has_hardhat: bool = False
    has_vest: bool = False
    helmet_violation: bool = False   # se detectó NO-Hardhat
    vest_violation: bool = False     # se detectó NO-Safety Vest
    missing: list = field(default_factory=list)  # qué EPP falta

    @property
    def compliant(self) -> bool:
        return (
            self.has_hardhat
            and self.has_vest
            and not self.helmet_violation
            and not self.vest_violation
        )


def _center(box: tuple) -> tuple:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _area(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _point_in_box(point: tuple, box: tuple) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


class PPEComplianceChecker:
    """Aplica la regla de cumplimiento sobre las detecciones de un fotograma."""

    def __init__(
        self,
        min_conf: float = 0.25,
        head_frac: float = 0.40,   # región de cabeza: 40% superior de la persona
        torso_top: float = 0.20,   # torso: desde 20%...
        torso_bot: float = 0.75,   # ...hasta 75% de la altura de la persona
    ):
        self.min_conf = min_conf
        self.head_frac = head_frac
        self.torso_top = torso_top
        self.torso_bot = torso_bot

    def _assign_person(self, item_center: tuple, persons: list) -> int:
        """Devuelve el índice de la persona cuya caja contiene el centro del EPP.
        Si varias lo contienen, elige la de menor área (la más específica/cercana)."""
        best_idx, best_area = -1, float("inf")
        for i, p in enumerate(persons):
            if _point_in_box(item_center, p.xyxy):
                a = _area(p.xyxy)
                if a < best_area:
                    best_idx, best_area = i, a
        return best_idx

    def _in_head_region(self, center: tuple, person_box: tuple) -> bool:
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        return y1 <= center[1] <= y1 + self.head_frac * h

    def _in_torso_region(self, center: tuple, person_box: tuple) -> bool:
        x1, y1, x2, y2 = person_box
        h = y2 - y1
        return y1 + self.torso_top * h <= center[1] <= y1 + self.torso_bot * h

    def check_frame(self, detections: list):
        """Analiza un fotograma.
        Args:
            detections: lista de Detection.
        Returns:
            (statuses, compliance_rate) — lista de PersonStatus y la tasa [0..1] o None
            si no hay personas.
        """
        dets = [d for d in detections if d.conf >= self.min_conf]
        persons = [d for d in dets if d.cls_name == CLS_PERSON]
        statuses = [PersonStatus(box=p.xyxy) for p in persons]

        for d in dets:
            if d.cls_name == CLS_PERSON:
                continue
            c = _center(d.xyxy)
            idx = self._assign_person(c, persons)
            if idx < 0:
                continue  # EPP no asociado a ninguna persona
            st = statuses[idx]
            pbox = persons[idx].xyxy
            if d.cls_name == CLS_HARDHAT and self._in_head_region(c, pbox):
                st.has_hardhat = True
            elif d.cls_name == CLS_NO_HARDHAT and self._in_head_region(c, pbox):
                st.helmet_violation = True
            elif d.cls_name == CLS_VEST and self._in_torso_region(c, pbox):
                st.has_vest = True
            elif d.cls_name == CLS_NO_VEST and self._in_torso_region(c, pbox):
                st.vest_violation = True

        # Anotar qué falta (para reportes)
        for st in statuses:
            if not st.has_hardhat:
                st.missing.append("casco")
            if not st.has_vest:
                st.missing.append("chaleco")

        if not statuses:
            return statuses, None
        n_ok = sum(1 for st in statuses if st.compliant)
        return statuses, n_ok / len(statuses)


def annotate_frame(image, statuses, compliance_rate):
    """Dibuja sobre la imagen: cajas verdes (cumple) / rojas (no cumple) + banner con la tasa.

    Requiere OpenCV. `image` es un array BGR (como lo entrega cv2.imread / frame de video).
    Devuelve la imagen anotada.
    """
    import cv2

    GREEN = (0, 180, 0)
    RED = (0, 0, 255)

    for st in statuses:
        x1, y1, x2, y2 = map(int, st.box)
        color = GREEN if st.compliant else RED
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if st.compliant:
            label = "OK"
        else:
            label = "NO: falta " + ",".join(st.missing) if st.missing else "NO conforme"
        cv2.putText(image, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Banner superior con la tasa de cumplimiento del frame
    n_total = len(statuses)
    n_ok = sum(1 for st in statuses if st.compliant)
    if compliance_rate is None:
        text = "Sin personas detectadas"
    else:
        text = f"Cumplimiento: {compliance_rate:.0%}  ({n_ok}/{n_total})"
    cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
    cv2.putText(image, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return image


# =====================================================================================
# Self-test: valida la lógica con cajas SINTÉTICAS (no necesita el modelo entrenado)
# =====================================================================================
def _self_test() -> None:
    checker = PPEComplianceChecker(min_conf=0.25)

    # Escenario: 2 personas.
    #  Persona A (x 0-100): tiene casco (arriba) y chaleco (torso) -> CUMPLE
    #  Persona B (x 200-300): tiene NO-Hardhat (arriba), sin chaleco -> NO CUMPLE
    dets = [
        Detection(CLS_PERSON, 0.9, (0, 0, 100, 400)),
        Detection(CLS_HARDHAT, 0.8, (30, 10, 70, 60)),       # cabeza de A
        Detection(CLS_VEST, 0.8, (20, 150, 80, 280)),        # torso de A
        Detection(CLS_PERSON, 0.9, (200, 0, 300, 400)),
        Detection(CLS_NO_HARDHAT, 0.8, (230, 10, 270, 60)),  # cabeza de B (violación)
    ]
    statuses, rate = checker.check_frame(dets)

    assert len(statuses) == 2, "Debe haber 2 personas"
    assert statuses[0].compliant is True, "Persona A debería CUMPLIR"
    assert statuses[1].compliant is False, "Persona B NO debería cumplir"
    assert statuses[1].helmet_violation is True, "Persona B tiene NO-Hardhat"
    assert abs(rate - 0.5) < 1e-9, f"Tasa esperada 0.5, obtenida {rate}"

    print("Self-test OK ✅")
    print(f"  Persona A -> cumple={statuses[0].compliant} (casco={statuses[0].has_hardhat}, chaleco={statuses[0].has_vest})")
    print(f"  Persona B -> cumple={statuses[1].compliant} (violación casco={statuses[1].helmet_violation}, falta={statuses[1].missing})")
    print(f"  Tasa de cumplimiento del frame: {rate:.0%}")


if __name__ == "__main__":
    _self_test()
