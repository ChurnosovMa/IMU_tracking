"""
Приём ДВУХ кватернионов с UART и визуализация:
  - Туловище: параллелепипед, ориентация = q_body (reference)
  - Рука: капсула ("колбаска"), закреплена в плече туловища,
          вращается ОТНОСИТЕЛЬНО туловища (q_rel = conj(q_body) * q_arm)

Установка зависимостей:
    pip install pyserial pygame PyOpenGL PyOpenGL_accelerate

Ожидаемый формат строки от сенсоров (ОДНА строка, 8 чисел через запятую):
    w_body,x_body,y_body,z_body,w_arm,x_arm,y_arm,z_arm\n

Если у тебя другой формат — поменяй функцию parse_line().

КАЛИБРОВКА:
    Поставь руку в "домашнюю" позу (например, опущена вдоль тела) и нажми ПРОБЕЛ
    в окне визуализации. С этого момента текущая поза считается нулевой, и рука
    в 3D будет двигаться ровно так же, как ты двигаешь её физически —
    независимо от того, как сенсоры на самом деле прикручены к телу/руке.
    Можно калиброваться повторно в любой момент (просто снова встань в
    домашнюю позу и нажми пробел).
"""

import threading
import time
import math

import serial
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *

# ======================= НАСТРОЙКИ UART =======================
SERIAL_PORT = "/dev/ttyACM0"  # поменяй на свой порт, например "COM3" на Windows
BAUDRATE = 115200
# ===============================================================

# ======================= НАСТРОЙКИ МОДЕЛИ (см. пояснения внизу про масштабирование) =======================
TORSO_SIZE = (2.0, 3.0, 1.0)     # (ширина X, высота Y, глубина Z) параллелепипеда-тела
SHOULDER_OFFSET = (1.0, 1.3, 0)  # точка плеча на поверхности тела (правое плечо, чуть ниже верхней грани), в локальных координатах тела
ARM_LENGTH = 3.0                 # длина "руки"-колбаски
ARM_RADIUS = 0.35                # радиус колбаски
ARM_AXIS = "y"                   # вдоль какой локальной оси тянется рука ДО поворота ("x", "y" или "z")
# "y" + знак ниже в axis_to_rotation() -> рука по умолчанию свисает ВНИЗ (домашняя поза).
# Если после калибровки (пробел) рука в состоянии покоя торчит не туда (влево/вправо/вверх) —
# меняй axis_to_rotation() ниже (там всего 3 варианта на выбор + можно подставить свои градусы).
# ============================================================================================================

quat_lock = threading.Lock()
current_body_quat = (1.0, 0.0, 0.0, 0.0)  # w, x, y, z
current_arm_quat = (1.0, 0.0, 0.0, 0.0)
running = True

# --------------------- Калибровка (zero pose) ---------------------
# Запоминаем кватернионы в момент калибровки (когда рука опущена в "домашней" позе).
# До первой калибровки используем identity — т.е. поведение как "без калибровки".
IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
cal_lock = threading.Lock()
body_quat_cal = IDENTITY_QUAT
arm_rel_quat_cal = IDENTITY_QUAT  # relative (arm относительно body) в момент калибровки
is_calibrated = False


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


def axis_angle_quat(axis, angle_deg):
    """Кватернион поворота на angle_deg (градусы) вокруг axis=(x,y,z)."""
    angle = math.radians(angle_deg)
    s = math.sin(angle / 2)
    x, y, z = axis
    return (math.cos(angle / 2), x * s, y * s, z * s)


def euler_to_quat(rx_deg, ry_deg, rz_deg):
    """Кватернион из углов Эйлера (градусы), порядок применения X, затем Y, затем Z.
    Удобно для подбора remap-а осей методом проб и ошибок."""
    qx = axis_angle_quat((1, 0, 0), rx_deg)
    qy = axis_angle_quat((0, 1, 0), ry_deg)
    qz = axis_angle_quat((0, 0, 1), rz_deg)
    return quat_multiply(qz, quat_multiply(qy, qx))


# --------------------- REMAP ОСЕЙ СЕНСОРА -> ОСИ OPENGL ---------------------
# У сенсора своя система координат (например Z-вверх), у OpenGL здесь Y-вверх,
# X-вправо, Z-к зрителю. Если при физическом повороте тела ВЛЕВО-ВПРАВО (yaw,
# вокруг вертикальной оси) объект в 3D крутится вокруг НЕ той оси (например
# кренится вместо того чтобы вращаться как юла) — значит нужно поменять
# AXIS_REMAP ниже.
#
# Как подобрать (метод проб и ошибок):
#   1. Поставь тело неподвижно, поверни его физически ТОЛЬКО вокруг
#      вертикальной оси (как будто поворачиваешься на месте).
#   2. Смотри на экран: параллелепипед должен вращаться вокруг СВОЕЙ
#      вертикальной оси (как юла/волчок), никаких наклонов.
#   3. Если крутится не так — меняй числа ниже (обычно достаточно 90/-90/180
#      по одной из осей) и перезапускай, пока поворот не станет верным.
#   4. Дальше жми ПРОБЕЛ (калибровка), чтобы обнулить оставшееся смещение позы.
AXIS_REMAP = euler_to_quat(rx_deg=-90, ry_deg=90, rz_deg=0)  # <-- подбирай эти три числа


def apply_axis_remap(q_raw):
    """Меняет систему координат кватерниона: сенсор -> мир OpenGL (сходство/conjugation)."""
    return quat_multiply(quat_multiply(AXIS_REMAP, q_raw), quat_conjugate(AXIS_REMAP))


def calibrate(q_body, q_arm):
    """Фиксирует текущую позу как 'домашнюю' (ноль)."""
    global body_quat_cal, arm_rel_quat_cal, is_calibrated
    q_rel_now = quat_multiply(quat_conjugate(q_body), q_arm)
    with cal_lock:
        body_quat_cal = q_body
        arm_rel_quat_cal = q_rel_now
        is_calibrated = True
    print("Калибровка выполнена: текущая поза = домашняя (0).")


def quat_to_matrix(q):
    """Кватернион -> матрица 4x4 для glMultMatrixf (column-major)."""
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


# --------------------- Чтение UART ---------------------

def parse_line(line: str):
    """Парсит строку '8 чисел через запятую' -> (q_body, q_arm) или None."""
    try:
        parts = [float(p) for p in line.strip().split(" ")]
        if len(parts) != 8:
            return None
        q_body = tuple(parts[0:4])
        q_arm = tuple(parts[4:8])
        return q_body, q_arm
    except ValueError:
        return None


def serial_reader_thread():
    global current_body_quat, current_arm_quat, running

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
                q_body, q_arm = parsed
                q_body = apply_axis_remap(q_body)
                q_arm = apply_axis_remap(q_arm)
                with quat_lock:
                    current_body_quat = q_body
                    current_arm_quat = q_arm
        except Exception as e:
            print(f"Ошибка чтения serial: {e}")
            time.sleep(0.1)

    ser.close()


# --------------------- Рисование геометрии ---------------------

def draw_box(size):
    """Параллелепипед с центром в (0,0,0), size = (sx, sy, sz)."""
    sx, sy, sz = (s / 2.0 for s in size)
    vertices = (
        (sx, -sy, -sz), (sx, sy, -sz), (-sx, sy, -sz), (-sx, -sy, -sz),
        (sx, -sy, sz), (sx, sy, sz), (-sx, -sy, sz), (-sx, sy, sz),
    )
    faces = (
        (0, 1, 2, 3), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 3, 6, 7), (1, 2, 7, 5), (0, 3, 6, 4),
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 7), (3, 6),
    )
    colors = (
        (0.3, 0.5, 0.8), (0.3, 0.5, 0.8), (0.35, 0.55, 0.85),
        (0.35, 0.55, 0.85), (0.4, 0.6, 0.9), (0.4, 0.6, 0.9),
    )

    glBegin(GL_QUADS)
    for face, color in zip(faces, colors):
        glColor3fv(color)
        for idx in face:
            glVertex3fv(vertices[idx])
    glEnd()

    glColor3f(0.05, 0.05, 0.05)
    glBegin(GL_LINES)
    for edge in edges:
        for idx in edge:
            glVertex3fv(vertices[idx])
    glEnd()


def draw_capsule(length, radius):
    """
    'Колбаска' вдоль локальной оси X: цилиндр + 2 полусферы на концах.
    GLU рисует цилиндр/сферу вдоль оси Z, поэтому один раз разворачиваем на -90° по Y,
    чтобы получить ориентацию вдоль X.
    """
    quad = gluNewQuadric()
    gluQuadricNormals(quad, GLU_SMOOTH)

    glColor3f(0.85, 0.3, 0.25)

    glPushMatrix()
    glRotatef(-90, 0, 1, 0)  # ось Z -> ось X

    # сфера у основания (плечо)
    gluSphere(quad, radius, 16, 16)

    # цилиндр вдоль оси
    gluCylinder(quad, radius, radius, length, 16, 4)

    # сфера на конце (кисть/локоть)
    glTranslatef(0, 0, length)
    gluSphere(quad, radius, 16, 16)

    glPopMatrix()
    gluDeleteQuadric(quad)


def axis_to_rotation(axis):
    """Доп. поворот, чтобы капсула 'смотрела' вдоль нужной локальной оси тела в состоянии покоя (joint = identity)."""
    if axis == "x":
        return None                # вдоль +X (обычно "вбок")
    elif axis == "y":
        return (90, 0, 0, 1)      # вдоль -Y (ВНИЗ) -- если торчит вверх, поставь (90, 0, 0, 1)
    elif axis == "z":
        return (-90, 0, 1, 0)      # вдоль +Z/-Z ("вперёд/назад")
    return None


# --------------------- Основной цикл ---------------------

def main():
    global running

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (900, 700)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Body + Arm Quaternion Viewer  |  SPACE = calibrate (zero pose), ESC = exit")

    gluPerspective(45, (display[0] / display[1]), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -14)  # камера дальше, т.к. модель крупнее куба
    glEnable(GL_DEPTH_TEST)

    clock = pygame.time.Clock()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    with quat_lock:
                        q_body_snapshot = current_body_quat
                        q_arm_snapshot = current_arm_quat
                    calibrate(q_body_snapshot, q_arm_snapshot)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            with quat_lock:
                q_body = current_body_quat
                q_arm = current_arm_quat

            with cal_lock:
                q_body_cal = body_quat_cal
                q_rel_cal = arm_rel_quat_cal

            # насколько тело повернулось С МОМЕНТА калибровки (мировой дельта-поворот тела)
            q_body_delta = quat_multiply(q_body, quat_conjugate(q_body_cal))

            # рука относительно тела ПРЯМО СЕЙЧАС
            q_rel_instant = quat_multiply(quat_conjugate(q_body), q_arm)

            # чистое движение руки относительно тела, за вычетом позы/оффсета в момент калибровки
            q_joint = quat_multiply(q_rel_instant, quat_conjugate(q_rel_cal))

            body_matrix = quat_to_matrix(q_body_delta)
            rel_matrix = quat_to_matrix(q_joint)

            # --- Тело ---
            glPushMatrix()
            glMultMatrixf(body_matrix)
            draw_box(TORSO_SIZE)
            glPopMatrix()

            # --- Рука: сначала трансформ тела (чтобы плечо "ехало" вместе с телом),
            #     потом сдвиг в точку плеча, потом относительный поворот руки ---
            glPushMatrix()
            glMultMatrixf(body_matrix)
            glTranslatef(*SHOULDER_OFFSET)
            glMultMatrixf(rel_matrix)

            extra_rot = axis_to_rotation(ARM_AXIS)
            if extra_rot is not None:
                glRotatef(*extra_rot)

            draw_capsule(ARM_LENGTH, ARM_RADIUS)
            glPopMatrix()

            pygame.display.flip()
            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()