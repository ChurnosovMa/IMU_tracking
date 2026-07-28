"""
Приём ДВУХ кватернионов с UART и визуализация модели "плечо + предплечье"
со свободной 3D-камерой (как в CAD/компасах).

Установка зависимостей:
    pip install pyserial pygame PyOpenGL PyOpenGL_accelerate

Формат строки от MCU (ОДНА строка, 8 чисел через пробел):
    w_upper x_upper y_upper z_upper w_fore x_fore y_fore z_fore\n

УПРАВЛЕНИЕ КАМЕРОЙ:
    ЛКМ + перетаскивание   -> вращать вид вокруг модели (орбита)
    ПКМ + перетаскивание   -> панорамировать (двигать точку, вокруг которой вращаемся)
    Колесо мыши            -> зум (приближение/отдаление)
    Клавиша 1               -> вид спереди
    Клавиша 2               -> вид сбоку
    Клавиша 3               -> вид сверху
    Клавиша 0               -> сброс камеры в стандартный вид
    ПРОБЕЛ                  -> калибровка (текущая поза = "рука прямая" = ноль)
    ESC                     -> выход

ОСИ (гизмо в центре сцены):
    X - красный, Y - зелёный (вверх), Z - синий
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
# ============================================================================================================

quat_lock = threading.Lock()
current_upper_quat = (1.0, 0.0, 0.0, 0.0)
current_fore_quat = (1.0, 0.0, 0.0, 0.0)
running = True

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)
cal_lock = threading.Lock()
upper_quat_cal = IDENTITY_QUAT
fore_rel_quat_cal = IDENTITY_QUAT
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
    global upper_quat_cal, fore_rel_quat_cal, is_calibrated
    q_rel_now = quat_multiply(quat_conjugate(q_upper), q_fore)
    with cal_lock:
        upper_quat_cal = q_upper
        fore_rel_quat_cal = q_rel_now
        is_calibrated = True
    print("Калибровка выполнена: текущая поза (рука прямая) = домашняя (0).")


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
        except Exception as e:
            print(f"Ошибка чтения serial: {e}")
            time.sleep(0.1)

    ser.close()


# --------------------- Векторная математика (для камеры) ---------------------

def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a):
    length = math.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if length < 1e-8:
        return (0.0, 0.0, 0.0)
    return (a[0] / length, a[1] / length, a[2] / length)


# --------------------- Камера (орбита вокруг точки target) ---------------------

class OrbitCamera:
    def __init__(self):
        self.reset()

    def reset(self):
        self.azimuth = 45.0     # градусы, поворот вокруг вертикали
        self.elevation = 20.0   # градусы, наклон вверх/вниз
        self.distance = 14.0
        self.target = [0.0, -1.0, 0.0]  # немного ниже центра, т.к. рука "свисает" вниз

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
        """Возвращает (forward, right, up) единичные векторы текущей камеры."""
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
    """Гизмо осей в начале координат: X-красный, Y-зелёный, Z-синий."""
    glLineWidth(3.0)
    glBegin(GL_LINES)
    glColor3f(1, 0, 0); glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
    glColor3f(0, 1, 0); glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
    glColor3f(0.2, 0.4, 1); glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
    glEnd()
    glLineWidth(1.0)


def draw_floor_grid(size=8, step=1):
    """Сетка на плоскости XZ (y = -3), чтобы было видно 'низ' и масштаб."""
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


# --------------------- Основной цикл ---------------------

def main():
    global running

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (1000, 750)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(
        "Upper Arm + Forearm Viewer  |  ЛКМ вращать, ПКМ панорама, колесо зум, "
        "1/2/3 - виды, 0 - сброс, SPACE - калибровка, ESC - выход"
    )

    # ---------- ВАЖНО: матрица проекции ----------
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, (display[0] / display[1]), 0.1, 200.0)
    glMatrixMode(GL_MODELVIEW)
    # ---------------------------------------------
    glEnable(GL_DEPTH_TEST)

    cam = OrbitCamera()
    dragging_rotate = False
    dragging_pan = False

    clock = pygame.time.Clock()

    print("Управление: ЛКМ - вращать камеру, ПКМ - панорама, колесо - зум,")
    print("клавиши 1/2/3 - быстрые виды (спереди/сбоку/сверху), 0 - сброс камеры,")
    print("SPACE - калибровка, ESC - выход.")

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
                    elif event.key == pygame.K_1:
                        cam.set_view(azimuth=0.0, elevation=0.0)      # спереди
                    elif event.key == pygame.K_2:
                        cam.set_view(azimuth=90.0, elevation=0.0)     # сбоку
                    elif event.key == pygame.K_3:
                        cam.set_view(azimuth=0.0, elevation=89.0)     # сверху
                    elif event.key == pygame.K_0:
                        cam.reset()

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging_rotate = True
                    elif event.button == 3:
                        dragging_pan = True
                    elif event.button == 4:   # колесо вверх
                        cam.zoom(1.0)
                    elif event.button == 5:   # колесо вниз
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

            with quat_lock:
                q_upper = current_upper_quat
                q_fore = current_fore_quat

            with cal_lock:
                q_upper_cal = upper_quat_cal
                q_rel_cal = fore_rel_quat_cal

            q_upper_delta = quat_multiply(q_upper, quat_conjugate(q_upper_cal))
            q_rel_instant = quat_multiply(quat_conjugate(q_upper), q_fore)
            q_joint = quat_multiply(q_rel_instant, quat_conjugate(q_rel_cal))

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
            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()