"""
Приём ДВУХ кватернионов с UART и визуализация модели "плечо + предплечье".

ВЕРСИЯ С ОДНОЙ МИНИМАЛЬНОЙ КАЛИБРОВКОЙ (поза "рука вниз").

Идея: датчик даёт АБСОЛЮТНУЮ ориентацию (кватернион уже привязан к гравитации
через встроенный фьюжн MPU6050), но ось, вдоль которой физически направлена
кость, В СОБСТВЕННЫХ координатах датчика зависит от того, КАК ИМЕННО датчик
приклеен к руке - заранее это направление неизвестно и константой не
задаётся. Поэтому раньше "рука вниз" не давала 0° - мы предполагали не то
направление.

Решение - одна простая калибровка: встать в позу "рука вдоль тела, вниз" и
ЗАМЕРИТЬ, куда в координатах ДАТЧИКА сейчас "смотрит" гравитация. Это
направление (в системе датчика) и есть искомая продольная ось кости - именно
её нужно поворачивать кватернионом на каждом кадре, чтобы получать текущее
направление кости в мире. Никакого другого измерения (движения, PCA и т.п.)
не требуется - это буквально одна статичная поза, один снятый вектор на
каждый датчик.

Нам сейчас всё ещё не нужен угол прецессии (поворот плоскости движения руки
вокруг вертикали - придёт с 3-м датчиком на груди). Нужен только УГОЛ
НУТАЦИИ - насколько рука отклонена от вертикали.

ВАЖНО (исправленная версия): нутация считается КАК УГОЛ МЕЖДУ НАПРАВЛЕНИЕМ
КОСТИ И ВЕРТИКАЛЬЮ НАПРЯМУЮ (через arccos скалярного произведения), БЕЗ
проекции на какую-либо плоскость. Раньше здесь была математическая ошибка:
угол считался через проекцию направления кости на плоскость Y-Z (с
отбрасыванием компоненты по оси X) и потом через atan2 - но вращение вокруг
вертикали (Y) физически ПЕРЕМЕШИВАЕТ компоненты X и Z, поэтому попытка
"убрать только X" не может быть инвариантна к повороту тела вокруг себя:
часть настоящего движения руки терялась, а часть паразитного поворота тела
добавлялась в результат. Отсюда и "рука гнётся сама по себе при повороте
тела" и "прямая рука выглядит согнутой".

arccos-формула лишена этой проблемы: угол вектора К ОСИ ВРАЩЕНИЯ не может
измениться от вращения вокруг этой же оси - это просто такое свойство
вращений, не требует никаких оговорок или доп. условий. Плата за это -
у нутации нет знака (0°..180°, без "стороны"): куда именно повёрнута рука
(вперёд/назад/влево/вправо) при одной и той же нутации не различить - это
работа будущего 3-го датчика (прецессия).

Порядок расчёта на каждом кадре:

  1) Берём "продольную ось кости" В СОБСТВЕННОЙ (уже переразмеченной через
     AXIS_REMAP) системе координат датчика - НЕ константу, а то, что
     ЗАМЕРЕНО калибровкой (см. выше).
  2) Поворачиваем эту ось кватернионом датчика - получаем, куда кость
     СЕЙЧАС направлена в МИРОВЫХ координатах.
  3) Угол между результатом и направлением "вниз" (0,-1,0), через
     arccos(скалярное произведение) - это и есть нутация. Благодаря
     калибровке из шага 1 - в самой позе калибровки этот угол честно
     равен 0°.

  Угол плеча = это же самое для датчика на плече.
  Угол предплечья = это же самое для датчика на предплечье (тоже абсолютный,
      относительно вертикали, а не относительно плеча!).
  Угол в локте = угол_предплечья - угол_плеча (разность соседних абсолютных
      углов - работает, ПОКА оба сегмента действительно находятся в одной
      плоскости относительно тела, т.е. пока нет независимого движения
      "вбок" одного сегмента без другого).

Управление калибровкой:
    G  -> старт/стоп записи позы "рука вниз, локоть прямой" (для ОБОИХ
          датчиков сразу - постоять неподвижно ~2-3 сек, G ещё раз стоп)

Когда добавите 3-й датчик (на груди): он даст угол прецессии - куда именно
(в какую сторону) сейчас "смотрит" плоскость движения руки относительно
тела. Сама нутация (этот файл) не изменится - прецессия дополнит её, а не
заменит.

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
UPPER_ARM_AXIS = "y"     # локальная ось датчика, вдоль которой направлена кость плеча
UPPER_ARM_SIGN = -1

FOREARM_LENGTH = 2.6
FOREARM_RADIUS = 0.32
FOREARM_AXIS = "y"       # локальная ось датчика, вдоль которой направлено предплечье
FOREARM_SIGN = -1

# Ось, вокруг которой мы поворачиваем 3D-модель на экране (ЧИСТО для
# отрисовки - к самому расчёту угла нутации отношения больше не имеет,
# см. объяснение выше и функцию nutation_angle_deg ниже). Раньше эта ось
# ошибочно использовалась и в самом расчёте угла - это и было источником
# бага "поворот тела меняет позу руки на экране": вращение вокруг Y
# (вертикали) математически ПЕРЕМЕШИВАЕТ компоненты X и Z, поэтому попытка
# "убрать X и мерить угол в Y-Z" не может быть инвариантна к вращению вокруг
# Y - часть настоящего движения руки утекала в X и терялась, а часть
# паразитного вращения тела утекала в Z и добавлялась в угол. Правильная
# нутация (см. ниже) вообще не использует эту ось для расчёта - только
# для того, чтобы было вокруг чего вращать модель на экране.
RENDER_AXIS = (1.0, 0.0, 0.0)
# ==================================================================

quat_lock = threading.Lock()
current_upper_quat = (1.0, 0.0, 0.0, 0.0)
current_fore_quat = (1.0, 0.0, 0.0, 0.0)
running = True

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

# --- калибровка "рука вниз": продольная ось кости В КООРДИНАТАХ ДАТЧИКА ---
# Заполняется один раз через клавишу G (см. try_compute_calibration ниже).
# До калибровки - None, нутацию считать нельзя (программа явно это покажет).
cal_lock = threading.Lock()
local_bone_axis_upper = None
local_bone_axis_fore = None
is_calibrated = False

CAPTURE_NONE, CAPTURE_DOWN = range(2)
capture_mode = CAPTURE_NONE
buf_upper = []
buf_fore = []


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


def rotate_vector_by_quat(q, v):
    """Повернуть вектор v (заданный в локальной СК датчика) кватернионом q -
    получаем этот же вектор, но выраженный в МИРОВОЙ СК."""
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


# Переразметка осей сырого кватерниона датчика в удобную для нас СК
# (как и в самой первой версии скрипта - физическая ориентация платы датчика
# на теле не совпадает с осями, в которых нам удобно считать, этот поворот
# компенсирует разницу один раз и одинаково для всех кадров).
AXIS_REMAP = euler_to_quat(rx_deg=-90, ry_deg=90, rz_deg=0)


def apply_axis_remap(q_raw):
    return quat_multiply(quat_multiply(AXIS_REMAP, q_raw), quat_conjugate(AXIS_REMAP))


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


# --------------------- Калибровка "рука вниз" + угол нутации ---------------------

# Направление "рука висит вниз" в мировых координатах - точка отсчёта (0°)
# для угла нутации.
DOWN_WORLD = (0.0, -1.0, 0.0)


def gravity_in_sensor_frame(q_sensor_to_world):
    """Направление 'вниз' (0,-1,0) в МИРОВОЙ СК, выраженное В КООРДИНАТАХ
    ДАТЧИКА. Ровно это направление в позе 'рука вниз, локоть прямой' и есть
    продольная ось кости, В КООРДИНАТАХ ЭТОГО КОНКРЕТНОГО ДАТЧИКА (зависит от
    того, как именно он приклеен - поэтому и нужна калибровка, не константа)."""
    q_world_to_sensor = quat_conjugate(q_sensor_to_world)
    return rotate_vector_by_quat(q_world_to_sensor, DOWN_WORLD)


def start_stop_capture():
    global capture_mode, buf_upper, buf_fore
    if capture_mode == CAPTURE_NONE:
        capture_mode = CAPTURE_DOWN
        buf_upper = []
        buf_fore = []
        print("[рука вниз] запись начата — встаньте прямо, рука вдоль тела, "
              "локоть прямой, постойте неподвижно ~2-3 сек...")
        return
    capture_mode = CAPTURE_NONE
    global local_bone_axis_upper, local_bone_axis_fore, is_calibrated
    if len(buf_upper) < 5 or len(buf_fore) < 5:
        print("Слишком мало сэмплов, повторите.")
        return
    g_up = np.mean([gravity_in_sensor_frame(q) for q in buf_upper], axis=0)
    g_fo = np.mean([gravity_in_sensor_frame(q) for q in buf_fore], axis=0)
    with cal_lock:
        local_bone_axis_upper = v_norm(tuple(g_up))
        local_bone_axis_fore = v_norm(tuple(g_fo))
        is_calibrated = True
    print(f"Калибровка готова. Ось плеча в СК датчика: "
          f"{tuple(round(c, 3) for c in local_bone_axis_upper)}")
    print(f"Ось предплечья в СК датчика: "
          f"{tuple(round(c, 3) for c in local_bone_axis_fore)}")


def nutation_angle_deg(q, local_bone_axis):
    """
    q               - сырой (уже переразмеченный) кватернион датчика
    local_bone_axis - продольная ось кости В ЛОКАЛЬНОЙ СК ДАТЧИКА, полученная
                       калибровкой "рука вниз" (см. gravity_in_sensor_frame
                       выше) - НЕ константа, у каждого датчика своя.

    НУТАЦИЯ = угол между текущим направлением кости и направлением "вниз",
    СЧИТАЕТСЯ НАПРЯМУЮ (arccos скалярного произведения), БЕЗ проекции на
    какую-либо плоскость. Диапазон 0°..180°, БЕЗ ЗНАКА - у нутации в
    классическом (гироскопическом) смысле знака и не бывает, это угол
    отклонения от полюса (вертикали), а не угол в конкретной плоскости.

    Ключевое свойство этой формулы: она НЕ МЕНЯЕТСЯ при повороте всего тела
    вокруг вертикали (прецессии) - вращение вокруг оси Y математически не
    может изменить угол вектора К ЭТОЙ ЖЕ ОСИ Y. Именно поэтому раньше
    (когда угол считался через проекцию на плоскость Y-Z и atan2) поворот
    тела портил картинку - та формула была не инвариантна к прецессии,
    хотя должна была быть. Эта - инвариантна по построению, без всяких
    дополнительных условий.

    Плата за простоту: у угла нет знака и нет "стороны" (раскрыть, куда
    именно повёрнута рука - вперёд, назад, влево, вправо - можно будет
    только после 3-го датчика, который определит прецессию). Сейчас мы
    сознательно этого не различаем.
    """
    world_dir = rotate_vector_by_quat(q, local_bone_axis)
    cos_theta = v_dot(world_dir, DOWN_WORLD)
    cos_theta = max(-1.0, min(1.0, cos_theta))  # защита от погрешностей округления
    return math.degrees(math.acos(cos_theta))


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


# --------------------- Общий расчёт углов ---------------------

def compute_current_angles():
    """Возвращает (shoulder_deg, elbow_deg) - None, None, если калибровка
    "рука вниз" (клавиша G) ещё не выполнена."""
    with cal_lock:
        if not is_calibrated:
            return None, None
        upper_local_axis = local_bone_axis_upper
        fore_local_axis = local_bone_axis_fore

    with quat_lock:
        q_upper = current_upper_quat
        q_fore = current_fore_quat

    shoulder_deg = nutation_angle_deg(q_upper, upper_local_axis)
    forearm_abs_deg = nutation_angle_deg(q_fore, fore_local_axis)

    # угол в локте = разница соседних АБСОЛЮТНЫХ углов (оба отсчитаны от одной
    # и той же вертикали в одной и той же плоскости - см. докстринг вверху файла)
    elbow_deg = forearm_abs_deg - shoulder_deg

    return shoulder_deg, elbow_deg


# --------------------- Основной цикл ---------------------

def main():
    global running

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (1000, 750)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    base_caption = "Nutation Viewer | G-калибровка(рука вниз) 1/2/3-виды 0-камера ESC-выход"
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

    print("Перед началом работы: G — встаньте прямо, рука вдоль тела, локоть")
    print("прямой, постойте неподвижно ~2-3 сек, потом G ещё раз — стоп.")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_g:
                        start_stop_capture()
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

            shoulder_deg, elbow_deg = compute_current_angles()

            # пока калибровка (G) не выполнена - рисуем модель в нулевой позе
            # и явно показываем это в заголовке окна, а не тихо считаем как 0°
            render_shoulder = shoulder_deg if shoulder_deg is not None else 0.0
            render_elbow = elbow_deg if elbow_deg is not None else 0.0

            # рендерим оба угла как повороты вокруг ОДНОЙ и той же оси - это
            # ЧИСТО для отрисовки на экране (см. RENDER_AXIS выше), к расчёту
            # самого угла (nutation_angle_deg) эта ось больше не имеет
            # отношения
            q_upper_render = axis_angle_quat(RENDER_AXIS, render_shoulder)
            q_joint_render = axis_angle_quat(RENDER_AXIS, render_elbow)

            upper_matrix = quat_to_matrix(q_upper_render)
            joint_matrix = quat_to_matrix(q_joint_render)

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
                if shoulder_deg is not None:
                    pygame.display.set_caption(
                        f"{base_caption} | плечо:{shoulder_deg:6.1f} локоть:{elbow_deg:6.1f}"
                    )
                else:
                    pygame.display.set_caption(f"{base_caption} | НЕ ОТКАЛИБРОВАНО (нажмите G)")

            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()