"""
Приём ДВУХ кватернионов с UART и визуализация модели "плечо + предплечье"
со свободной 3D-камерой.

БАЗА: ваша версия с исправленным знаком оси (axis_from_total_rotation) и
синхронизацией фреймов (q_upper_cal_raw / q_fore_cal_raw), БЕЗ клампинга угла
локтя — вернулись к ней, т.к. клампинг не решал проблему "сгибания при
поднятии прямой руки" (он и не мог: он только обрезает диапазон, а не лечит
причину рассогласования).

ДОБАВЛЕНО: диагностика для поиска причины "сгибания" при поднятии прямой руки.

  - В заголовок окна теперь ВСЕГДА (при включённой функц. калибровке) выводится:
      flex   - угол сгибания локтя (180 = прямая рука), БЕЗ клампинга
      swing  - "паразитное" отклонение вне оси сгибания (в идеале ~0 всегда,
               для истинного шарнира). Если swing растёт вместе с подъёмом
               плеча - это явный признак рассинхронизации/наводки между
               датчиками, а не реальное движение в локте.
      shoulder - на сколько градусов текущая поза плеча отличается от позы
               калибровки (грубый индикатор "насколько поднята рука").

  - Клавиша P -> печатает пронумерованный снимок в консоль (flex, swing,
    shoulder, а также сырые кватернионы). Используйте так:
        1) держите руку прямо ВНИЗ (поза калибровки) -> P
        2) прямая рука ВПЕРЁД -> P
        3) прямая рука В СТОРОНУ (вправо) -> P
        4) прямая рука ПОДНЯТА НА ~90° -> P
    Если flex остаётся близко к 180 на всех четырёх точках - калибровка в
    порядке, и дело в динамике движения (см. ниже). Если flex ощутимо и
    ПОВТОРЯЕМО падает с ростом shoulder - это систематическая ошибка калибровки
    осей, а не шум/рассинхрон.

ВАЖНО ПРО ФИЗИКУ: если локоть держится идеально жёстко (не сгибается на самом
деле), относительный угол между сегментами МАТЕМАТИЧЕСКИ обязан оставаться
постоянным при любом общем повороте руки в плече - калибровочные ошибки в
общий поворот не проникают (они сокращаются при вычитании ориентаций). Поэтому
если угол всё равно "гуляет" при поднятии прямой руки - ищите причину не в
самой калибровке осей, а в одном из двух мест:
  (a) сам локоть в реальности не абсолютно неподвижен при подъёме (это очень
      легко упустить - большинство людей не держат руку идеально прямой без
      сознательного усилия);
  (b) два датчика читаются с рассинхроном по времени на стороне MCU (если
      цикл опроса читает сначала один MPU6050, потом другой, при быстром
      движении между чтениями пройдёт время, и получится кажущийся "изгиб",
      пропорциональный скорости движения - должен пропадать, если ОСТАНОВИТЬ
      руку и подождать).
Так что стоит проверить отдельно: поднимаете руку быстро -> изгиб есть;
останавливаетесь и ждёте секунду -> изгиб исчезает? Если исчезает - это (b).
Если сохраняется в статике - это (a) или систематическая ошибка осей (тогда
сработает тест с 4 референсными позами выше).

Установка зависимостей:
    pip install pyserial pygame PyOpenGL PyOpenGL_accelerate numpy
"""

import threading
import time
import math

import numpy as np
import serial
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *

# ======================= НАСТРОЙКИ UART =======================
SERIAL_PORT = "COM15"
BAUDRATE = 115200
# ===============================================================

# ======================= НАСТРОЙКИ МОДЕЛИ =======================
UPPER_ARM_LENGTH = 3.0
UPPER_ARM_RADIUS = 0.4
UPPER_ARM_AXIS = "y"
UPPER_ARM_SIGN = -1

FOREARM_LENGTH = 2.6
FOREARM_RADIUS = 0.32
FOREARM_AXIS = "y"
FOREARM_SIGN = -1
# ==================================================================

quat_lock = threading.Lock()
current_upper_quat = (1.0, 0.0, 0.0, 0.0)
current_fore_quat = (1.0, 0.0, 0.0, 0.0)
running = True

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

# --- "поза = ноль" калибровка (сырые снимки датчиков в момент SPACE) ---
cal_lock = threading.Lock()
upper_quat_cal = IDENTITY_QUAT
fore_quat_cal = IDENTITY_QUAT
is_calibrated = False

# --- функциональная (анатомическая) калибровка ---
func_lock = threading.Lock()
seg2sensor_upper = IDENTITY_QUAT
seg2sensor_fore = IDENTITY_QUAT
functional_calibrated = False

CAPTURE_NONE, CAPTURE_GRAVITY, CAPTURE_SHOULDER, CAPTURE_ELBOW = range(4)
capture_mode = CAPTURE_NONE
buf_upper = []
buf_fore = []

gravity_upper_S = None
gravity_fore_S = None
shoulder_axis_upper_S = None
elbow_axis_fore_S = None

snapshot_counter = 0


# --------------------- Кватернионная математика ---------------------

def quat_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_normalize(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-9:
        return IDENTITY_QUAT
    return (w / n, x / n, y / n, z / n)


def rotate_vector_by_quat(q, v):
    qv = (0.0, v[0], v[1], v[2])
    r = quat_multiply(quat_multiply(q, qv), quat_conjugate(q))
    return (r[1], r[2], r[3])


def axis_angle_quat(axis, angle_deg):
    angle = math.radians(angle_deg)
    s = math.sin(angle / 2)
    x, y, z = axis
    return (math.cos(angle / 2), x * s, y * s, z * s)


def euler_to_quat(rx_deg, ry_deg, rz_deg):
    qx = axis_angle_quat((1, 0, 0), rx_deg)
    qy = axis_angle_quat((0, 1, 0), ry_deg)
    qz = axis_angle_quat((0, 0, 1), rz_deg)
    return quat_multiply(qz, quat_multiply(qy, qx))


AXIS_REMAP = euler_to_quat(rx_deg=-90, ry_deg=90, rz_deg=0)


def apply_axis_remap(q_raw):
    return quat_multiply(quat_multiply(AXIS_REMAP, q_raw), quat_conjugate(AXIS_REMAP))


def calibrate(q_upper, q_fore):
    global upper_quat_cal, fore_quat_cal, is_calibrated
    with cal_lock:
        upper_quat_cal = q_upper
        fore_quat_cal = q_fore
        is_calibrated = True
    print("Калибровка позы выполнена: текущая поза = домашняя (0).")


def quat_to_matrix(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-8:
        return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return [
        1 - (yy + zz), xy + wz, xz - wy, 0,
        xy - wz, 1 - (xx + zz), yz + wx, 0,
        xz + wy, yz - wx, 1 - (xx + yy), 0,
        0, 0, 0, 1,
    ]


def matrix3_to_quat(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m21 - m12) * s
        y = (m02 - m20) * s
        z = (m10 - m01) * s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return quat_normalize((w, x, y, z))


# --------------------- Функциональная калибровка ---------------------

def gravity_in_sensor_frame(q_sensor_to_world):
    q_world_to_sensor = quat_conjugate(q_sensor_to_world)
    return rotate_vector_by_quat(q_world_to_sensor, (0.0, -1.0, 0.0))


def angvel_axis_samples(quat_sequence):
    samples = []
    for i in range(len(quat_sequence) - 1):
        q1 = quat_sequence[i]
        q2 = quat_sequence[i + 1]
        dq = quat_multiply(quat_conjugate(q1), q2)
        w, x, y, z = dq
        if w < 0:
            x, y, z = -x, -y, -z
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 1e-4:
            samples.append((x / norm, y / norm, z / norm))
    return samples


def dominant_axis_line(vectors):
    if len(vectors) < 5:
        return None
    arr = np.array(vectors)
    cov = arr.T @ arr / len(arr)
    eigvals, eigvecs = np.linalg.eigh(cov)
    return tuple(eigvecs[:, np.argmax(eigvals)])


def axis_from_total_rotation(quat_sequence):
    if len(quat_sequence) < 2:
        return None
    q_start = quat_sequence[0]
    q_end = quat_sequence[-1]
    dq = quat_multiply(quat_conjugate(q_start), q_end)
    w, x, y, z = dq
    if w < 0:
        x, y, z = -x, -y, -z
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-4:
        return None
    return (x / norm, y / norm, z / norm)


def dominant_axis(quat_sequence):
    samples = angvel_axis_samples(quat_sequence)
    line = dominant_axis_line(samples)
    if line is None:
        return None
    ref = axis_from_total_rotation(quat_sequence)
    if ref is not None and v_dot(line, ref) < 0:
        line = tuple(-c for c in line)
    return line


def v_norm(a):
    n = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if n < 1e-8:
        return (0.0, 0.0, 0.0)
    return (a[0] / n, a[1] / n, a[2] / n)


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def build_segment_quat(axis_long_S, axis_flex_S):
    e_long = v_norm(axis_long_S)
    proj = v_dot(e_long, axis_flex_S)
    e_flex = v_norm(v_sub(axis_flex_S, v_scale(e_long, proj)))
    e_third = v_norm(v_cross(e_long, e_flex))
    return matrix3_to_quat(
        e_long[0], e_flex[0], e_third[0],
        e_long[1], e_flex[1], e_third[1],
        e_long[2], e_flex[2], e_third[2],
    )


def try_compute_functional_calibration():
    global seg2sensor_upper, seg2sensor_fore, functional_calibrated
    ok_upper = gravity_upper_S is not None and shoulder_axis_upper_S is not None
    ok_fore = gravity_fore_S is not None and elbow_axis_fore_S is not None
    if not (ok_upper and ok_fore):
        print("Не хватает данных: нужны обе статичные позы и оба движения "
              "(G для гравитации, F для плеча, B для локтя).")
        return
    with func_lock:
        seg2sensor_upper = build_segment_quat(gravity_upper_S, shoulder_axis_upper_S)
        seg2sensor_fore = build_segment_quat(gravity_fore_S, elbow_axis_fore_S)
        functional_calibrated = True
    print("Функциональная калибровка построена.")


def reset_functional_calibration():
    global functional_calibrated, gravity_upper_S, gravity_fore_S
    global shoulder_axis_upper_S, elbow_axis_fore_S
    with func_lock:
        functional_calibrated = False
    gravity_upper_S = None
    gravity_fore_S = None
    shoulder_axis_upper_S = None
    elbow_axis_fore_S = None
    print("Функциональная калибровка сброшена.")


def start_stop_capture(mode, name):
    global capture_mode, buf_upper, buf_fore
    global gravity_upper_S, gravity_fore_S, shoulder_axis_upper_S, elbow_axis_fore_S

    if capture_mode == CAPTURE_NONE:
        capture_mode = mode
        buf_upper = []
        buf_fore = []
        print(f"[{name}] запись начата — выполните позу/движение один раз...")
        return

    if capture_mode != mode:
        print("Сначала завершите текущую запись (нажмите ту же клавишу ещё раз).")
        return

    capture_mode = CAPTURE_NONE
    if mode == CAPTURE_GRAVITY:
        if len(buf_upper) < 5:
            print("Слишком мало сэмплов для гравитации, повторите.")
            return
        g_up = np.mean([gravity_in_sensor_frame(q) for q in buf_upper], axis=0)
        g_fo = np.mean([gravity_in_sensor_frame(q) for q in buf_fore], axis=0)
        gravity_upper_S = tuple(g_up)
        gravity_fore_S = tuple(g_fo)
        print(f"[{name}] готово: продольные оси захвачены.")
    elif mode == CAPTURE_SHOULDER:
        axis = dominant_axis(buf_upper)
        if axis is None:
            print("Слишком мало движения было зафиксировано, повторите увереннее.")
            return
        shoulder_axis_upper_S = axis
        print(f"[{name}] готово: ось плеча захвачена.")
    elif mode == CAPTURE_ELBOW:
        axis = dominant_axis(buf_fore)
        if axis is None:
            print("Слишком мало движения было зафиксировано, повторите увереннее.")
            return
        elbow_axis_fore_S = axis
        print(f"[{name}] готово: ось локтя захвачена.")


# --------------------- Диагностика (без клампинга!) ---------------------

def quat_angle_deg(q):
    """Величина поворота кватерниона от единичного (кратчайший путь), градусы."""
    w = max(-1.0, min(1.0, q[0]))
    return math.degrees(2.0 * math.acos(abs(w)))


def swing_twist_around_y(q):
    """Разложить на swing (всё, кроме вращения вокруг локальной Y) и twist
    (вращение вокруг локальной Y — оси сгибания по калибровке)."""
    w, x, y, z = q
    twist = quat_normalize((w, 0.0, y, 0.0))
    swing = quat_multiply(q, quat_conjugate(twist))
    return swing, twist


def twist_y_angle_deg(twist):
    w, x, y, z = twist
    return math.degrees(2.0 * math.atan2(y, w))


def elbow_diagnostics(q_joint):
    """(flex_angle_deg, swing_angle_deg) — БЕЗ клампинга, чисто для отладки.
    flex: 180 = прямая рука (условно, знак не подобран - смотрите тренд, не
    абсолютное число). swing: чем больше, тем сильнее локоть "гнётся" не
    вокруг калиброванной оси сгибания - в идеале должно быть ~0 всегда,
    рост swing вместе с подъёмом плеча = верный признак рассинхрона/наводки."""
    swing, twist = swing_twist_around_y(q_joint)
    theta_deg = twist_y_angle_deg(twist)
    flex_deg = 180.0 - theta_deg
    swing_deg = quat_angle_deg(swing)
    return flex_deg, swing_deg


# --------------------- Чтение UART ---------------------

def parse_line(line: str):
    try:
        parts = [float(p) for p in line.strip().split(" ")]
        if len(parts) != 8:
            return None
        q_upper = tuple(parts[0:4])
        q_fore = tuple(parts[4:8])
        return q_upper, q_fore
    except ValueError:
        return None


def serial_reader_thread():
    global current_upper_quat, current_fore_quat, running

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except serial.SerialException as e:
        print(f"Не удалось открыть порт {SERIAL_PORT}: {e}")
        running = False
        return

    print(f"Порт {SERIAL_PORT} открыт, жду данные...")

    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            parsed = parse_line(line)
            if parsed is not None:
                q_upper, q_fore = parsed
                q_upper = apply_axis_remap(q_upper)
                q_fore = apply_axis_remap(q_fore)
                with quat_lock:
                    current_upper_quat = q_upper
                    current_fore_quat = q_fore
                if capture_mode != CAPTURE_NONE:
                    buf_upper.append(q_upper)
                    buf_fore.append(q_fore)
        except Exception as e:
            print(f"Ошибка чтения serial: {e}")
            time.sleep(0.1)

    ser.close()


# --------------------- Камера ---------------------

class OrbitCamera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.azimuth = 45.0
        self.elevation = 20.0
        self.distance = 14.0
        self.target = [0.0, -1.0, 0.0]

    def set_view(self, azimuth, elevation):
        self.azimuth = azimuth
        self.elevation = elevation

    def position(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cx = self.target[0] + self.distance * math.cos(el) * math.sin(az)
        cy = self.target[1] + self.distance * math.sin(el)
        cz = self.target[2] + self.distance * math.cos(el) * math.cos(az)
        return (cx, cy, cz)

    def basis(self):
        pos = self.position()
        forward = v_norm(v_sub(tuple(self.target), pos))
        world_up = (0.0, 1.0, 0.0)
        right = v_norm(v_cross(forward, world_up))
        up = v_norm(v_cross(right, forward))
        return forward, right, up

    def rotate(self, dx_pixels, dy_pixels):
        self.azimuth += dx_pixels * 0.3
        self.elevation += dy_pixels * 0.3
        self.elevation = max(-89.0, min(89.0, self.elevation))

    def pan(self, dx_pixels, dy_pixels):
        _, right, up = self.basis()
        pan_speed = self.distance * 0.0025
        move = v_add(v_scale(right, -dx_pixels * pan_speed), v_scale(up, dy_pixels * pan_speed))
        self.target = list(v_add(tuple(self.target), move))

    def zoom(self, amount):
        self.distance = max(2.0, min(60.0, self.distance - amount))

    def apply(self):
        glLoadIdentity()
        pos = self.position()
        gluLookAt(pos[0], pos[1], pos[2],
                   self.target[0], self.target[1], self.target[2],
                   0, 1, 0)


# --------------------- Рисование геометрии ---------------------

AXIS_VECTORS = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


def draw_segment(length, radius, axis="y", sign=1):
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)
    glColor3f(0.85, 0.3, 0.25)
    glPushMatrix()
    if axis == "x":
        glRotatef(90 * sign, 0, 1, 0)
    elif axis == "y":
        glRotatef(-90 * sign, 1, 0, 0)
    elif axis == "z":
        if sign < 0:
            glRotatef(180, 1, 0, 0)
    gluSphere(quad, radius, 16, 16)
    gluCylinder(quad, radius, radius, length, 16, 4)
    glTranslatef(0, 0, length)
    gluSphere(quad, radius, 16, 16)
    glPopMatrix()
    gluDeleteQuadric(quad)


def segment_tip_offset(length, axis="y", sign=1):
    vx, vy, vz = AXIS_VECTORS[axis]
    return (vx * length * sign, vy * length * sign, vz * length * sign)


def draw_world_axes(length=2.5):
    glLineWidth(3.0)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
    glColor3f(0.2, 0.4, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
    glEnd()
    glLineWidth(1.0)


def draw_floor_grid(size=8, step=1):
    y = -3.0
    glColor3f(0.35, 0.35, 0.35)
    glBegin(GL_LINES)
    i = -size
    while i <= size:
        glVertex3f(i, y, -size)
        glVertex3f(i, y, size)
        glVertex3f(-size, y, i)
        glVertex3f(size, y, i)
        i += step
    glEnd()


# --------------------- Общий расчёт ориентации ---------------------

def compute_current_orientation():
    """Возвращает (q_upper_delta, q_joint, flex_deg_or_None, swing_deg_or_None,
    shoulder_deg_or_None). БЕЗ клампинга — числа только для отображения/отладки."""
    with quat_lock:
        q_upper = current_upper_quat
        q_fore = current_fore_quat

    with cal_lock:
        q_upper_cal_raw = upper_quat_cal
        q_fore_cal_raw = fore_quat_cal

    with func_lock:
        use_functional = functional_calibrated
        s2s_upper = seg2sensor_upper
        s2s_fore = seg2sensor_fore

    if use_functional:
        q_upper_seg_world = quat_multiply(q_upper, s2s_upper)
        q_fore_seg_world = quat_multiply(q_fore, s2s_fore)
        q_upper_cal_seg = quat_multiply(q_upper_cal_raw, s2s_upper)
        q_fore_cal_seg = quat_multiply(q_fore_cal_raw, s2s_fore)

        q_upper_delta = quat_multiply(q_upper_seg_world, quat_conjugate(q_upper_cal_seg))
        q_rel_instant = quat_multiply(quat_conjugate(q_upper_seg_world), q_fore_seg_world)
        q_rel_cal_seg = quat_multiply(quat_conjugate(q_upper_cal_seg), q_fore_cal_seg)
        q_joint = quat_multiply(q_rel_instant, quat_conjugate(q_rel_cal_seg))

        flex_deg, swing_deg = elbow_diagnostics(q_joint)
        shoulder_deg = quat_angle_deg(q_upper_delta)
        return q_upper_delta, q_joint, flex_deg, swing_deg, shoulder_deg
    else:
        q_rel_cal_raw = quat_multiply(quat_conjugate(q_upper_cal_raw), q_fore_cal_raw)
        q_upper_delta = quat_multiply(q_upper, quat_conjugate(q_upper_cal_raw))
        q_rel_instant = quat_multiply(quat_conjugate(q_upper), q_fore)
        q_joint = quat_multiply(q_rel_instant, quat_conjugate(q_rel_cal_raw))
        return q_upper_delta, q_joint, None, None, None


# --------------------- Основной цикл ---------------------

def main():
    global running, snapshot_counter

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (1000, 750)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    base_caption = ("Upper Arm + Forearm Viewer | G-гравитация F-плечо B-локоть "
                    "C-посчитать R-сброс SPACE-поза=0 P-снимок 1/2/3-виды 0-камера ESC-выход")
    pygame.display.set_caption(base_caption)

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_DEPTH_TEST)

    cam = OrbitCamera()
    dragging_rotate = False
    dragging_pan = False
    clock = pygame.time.Clock()
    caption_timer = 0.0

    print("Порядок функциональной калибровки:")
    print("  1) G — рука вдоль тела, локоть прямой, стоять неподвижно ~2-3 сек, G ещё раз стоп")
    print("  2) F — ОДНО чистое движение: локоть прямой, поднять прямую руку вперёд один раз, F стоп")
    print("  3) B — плечо неподвижно, ОДНО чистое сгибание локтя вперёд, B стоп")
    print("  4) C — построить калибровку. R — сбросить.")
    print("  SPACE — обнулить текущую позу (нажимать ПОСЛЕ C).")
    print("  P — снять диагностический снимок (для референсных поз, см. докстринг файла).")

    latest = None  # последний (q_upper_delta, q_joint, flex, swing, shoulder)

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        with quat_lock:
                            q_upper_snapshot = current_upper_quat
                            q_fore_snapshot = current_fore_quat
                        calibrate(q_upper_snapshot, q_fore_snapshot)
                    elif event.key == pygame.K_g:
                        start_stop_capture(CAPTURE_GRAVITY, "гравитация")
                    elif event.key == pygame.K_f:
                        start_stop_capture(CAPTURE_SHOULDER, "ось плеча")
                    elif event.key == pygame.K_b:
                        start_stop_capture(CAPTURE_ELBOW, "ось локтя")
                    elif event.key == pygame.K_c:
                        try_compute_functional_calibration()
                    elif event.key == pygame.K_r:
                        reset_functional_calibration()
                    elif event.key == pygame.K_p:
                        snapshot_counter += 1
                        if latest is not None:
                            _, _, flex_deg, swing_deg, shoulder_deg = latest
                            with quat_lock:
                                q_u, q_f = current_upper_quat, current_fore_quat
                            print(f"--- снимок #{snapshot_counter} ---")
                            print(f"  flex={flex_deg}  swing={swing_deg}  shoulder={shoulder_deg}")
                            print(f"  q_upper_raw={q_u}")
                            print(f"  q_fore_raw ={q_f}")
                        else:
                            print("Нет данных (включите функциональную калибровку сначала).")
                    elif event.key == pygame.K_1:
                        cam.set_view(azimuth=0.0, elevation=0.0)
                    elif event.key == pygame.K_2:
                        cam.set_view(azimuth=90.0, elevation=0.0)
                    elif event.key == pygame.K_3:
                        cam.set_view(azimuth=0.0, elevation=89.0)
                    elif event.key == pygame.K_0:
                        cam.reset()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging_rotate = True
                    elif event.button == 3:
                        dragging_pan = True
                    elif event.button == 4:
                        cam.zoom(1.0)
                    elif event.button == 5:
                        cam.zoom(-1.0)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging_rotate = False
                    elif event.button == 3:
                        dragging_pan = False
                elif event.type == pygame.MOUSEWHEEL:
                    cam.zoom(event.y * 1.0)
                elif event.type == pygame.MOUSEMOTION:
                    dx, dy = event.rel
                    if dragging_rotate:
                        cam.rotate(dx, dy)
                    elif dragging_pan:
                        cam.pan(dx, dy)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            cam.apply()
            draw_floor_grid()
            draw_world_axes()

            latest = compute_current_orientation()
            q_upper_delta, q_joint, flex_deg, swing_deg, shoulder_deg = latest

            upper_matrix = quat_to_matrix(q_upper_delta)
            joint_matrix = quat_to_matrix(q_joint)

            glPushMatrix()
            glMultMatrixf(upper_matrix)
            draw_segment(UPPER_ARM_LENGTH, UPPER_ARM_RADIUS, UPPER_ARM_AXIS, UPPER_ARM_SIGN)
            tip = segment_tip_offset(UPPER_ARM_LENGTH, UPPER_ARM_AXIS, UPPER_ARM_SIGN)
            glTranslatef(*tip)
            glMultMatrixf(joint_matrix)
            draw_segment(FOREARM_LENGTH, FOREARM_RADIUS, FOREARM_AXIS, FOREARM_SIGN)
            glPopMatrix()

            pygame.display.flip()

            caption_timer += clock.get_time() / 1000.0
            if caption_timer > 0.2:
                caption_timer = 0.0
                if flex_deg is not None:
                    pygame.display.set_caption(
                        f"{base_caption} | flex:{flex_deg:5.1f} swing:{swing_deg:5.1f} "
                        f"shoulder:{shoulder_deg:5.1f}"
                    )
                else:
                    pygame.display.set_caption(base_caption)

            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()