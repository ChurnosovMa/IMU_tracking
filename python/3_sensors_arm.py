"""
Приём ТРЁХ кватернионов с UART (плечо, предплечье, грудь) и визуализация
модели "тело (трапеция) + плечо + предплечье" со свободной 3D-камерой.

НОВОЕ В ЭТОЙ ВЕРСИИ: третий датчик - GY-BNO08x на груди, С МАГНИТОМЕТРОМ.
Он даёт АБСОЛЮТНУЮ, НЕ ДРЕЙФУЮЩУЮ во времени ориентацию (в отличие от
MPU6050 на руке, у которого без магнитометра рысканье/yaw дрейфует). Это
позволяет наконец определить УГОЛ ПРЕЦЕССИИ - куда сейчас "смотрит" плоскость,
в которой рука поднимается и сгибается, относительно тела.

============================================================================
СИСТЕМА КООРДИНАТ
============================================================================
Наша собственная (человека) система: Y - вперёд, X - влево, Z - вверх.
Ось Z (вертикаль) уже откалибрована раньше (через гравитацию, поза "рука
вниз") - это ось, вокруг которой и происходит прецессия.

Датчики плеча и предплечья (MPU6050) сами по себе НЕ знают, куда "вперёд" -
их наклон (roll/pitch) верный (за счёт гравитации), а вот рысканье (yaw)
привязано к СОБСТВЕННОЙ, ПРОИЗВОЛЬНОЙ точке отсчёта каждого датчика (и может
слегка дрейфовать за длинную сессию). Датчик груди (BNO08x) знает "вперёд"
АБСОЛЮТНО и стабильно (магнитометр).

============================================================================
ВТОРАЯ КАЛИБРОВКА: "РУКА ВПЕРЁД" (клавиша F)
============================================================================
Нужна ОДНА простая поза: рука вытянута вперёд (не важно, идеально горизонтально
или нет, не важен угол в локте - калибруется ТОЛЬКО направление/азимут).

В момент калибровки одновременно фиксируем:
  1) Направление кости плеча (из СОБСТВЕННЫХ показаний датчика плеча,
     используя уже откалиброванную ось кости - см. local_bone_axis_upper),
     спроецированное на горизонт. ЭТО направление объявляется осью Y_GROUP
     ("вперёд" для всей группы датчиков руки) - выражено в СОБСТВЕННЫХ
     координатах датчика плеча. Ключевой момент по вашей просьбе: ось
     берём ИМЕННО из датчика ПЛЕЧА, не предплечья.
  2) Показания датчика ГРУДИ в этот же момент (q_chest_cal) - это НАЧАЛО
     ОТСЧЁТА для отслеживания поворота ТЕЛА в дальнейшем.

============================================================================
РАСЧЁТ УГЛА ПРЕЦЕССИИ (на каждом кадре)
============================================================================
Угол прецессии = azimuth_arm(t) - body_yaw_since_cal(t) (ВЫЧИТАНИЕ, не
сложение! - см. пояснение ниже, это принципиально), где:

  azimuth_arm(t)         - насколько ТЕКУЩЕЕ направление кости плеча (в его
                            СОБСТВЕННЫХ координатах) отклонилось от Y_GROUP,
                            посчитанной при калибровке. ВАЖНО: собственная СК
                            датчика плеча - это ФИКСИРОВАННАЯ (инерциальная)
                            система отсчёта, она НЕ вращается вместе с телом.
                            Поэтому если ТЕЛО целиком повернётся, а рука
                            относительно тела не шевельнётся - датчик плеча
                            честно зарегистрирует это как реальный поворот
                            (он "поехал" вместе с телом) - то есть
                            azimuth_arm ВКЛЮЧАЕТ В СЕБЯ и поворот тела тоже,
                            не только независимое движение самой руки.

  body_yaw_since_cal(t)  - насколько ПОВЕРНУЛОСЬ ТЕЛО (грудь) с момента
                            калибровки, отдельно, ИЗ ДАТЧИКА ГРУДИ (магнитометр,
                            без дрейфа).

  Поскольку azimuth_arm УЖЕ содержит внутри себя поворот тела (см. выше), а
  body_yaw_since_cal - это ТОЧНО ТОТ ЖЕ поворот тела, посчитанный отдельно и
  точно - ВЫЧИТАНИЕ убирает его из azimuth_arm, оставляя ТОЛЬКО независимое
  движение руки ОТНОСИТЕЛЬНО тела. Именно это и нужно: чистый поворот тела
  (без движения руки относительно тела) должен давать результат ~0, и
  вычитание это обеспечивает, а сложение - нет (даёт двойной учёт одного и
  того же поворота, проверено численно перед тем как этот файл был отдан).

Итог: total_deg = azimuth_arm_deg - body_yaw_since_cal_deg - это и есть угол
поворота плоскости руки ОТНОСИТЕЛЬНО ТЕЛА, устойчивый к тому, что человек
крутится на месте (для чего мы и добавляли 3-й датчик), при этом сгибание
руки (нутация плеча, нутация предплечья, угол в локте) считается СОВЕРШЕННО
ТАК ЖЕ, как в предыдущей версии - это не изменилось и не должно было.

============================================================================
ОТОБРАЖЕНИЕ "ПО КОМПАСУ" (север/юг/запад/восток)
============================================================================
Это ОТДЕЛЬНАЯ, чисто информационная величина для вывода в консоль - не
влияет на расчёт углов руки выше. Требует знать, какая ЛОКАЛЬНАЯ ось датчика
груди указывает "вперёд, наружу от тела" - это CHEST_LOCAL_FORWARD_AXIS
ниже, ПОДБИРАЕТСЯ ЭМПИРИЧЕСКИ (встаньте лицом на север по компасу телефона,
проверьте что печатается ~0°, при необходимости смените ось/знак - точно
так же, как раньше подбирался AXIS_REMAP для датчиков руки).

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
# Формат строки: 12 чисел через пробел -
#   w_upper x_upper y_upper z_upper  w_fore x_fore y_fore z_fore  w_chest x_chest y_chest z_chest
# ===============================================================

# ======================= НАСТРОЙКИ МОДЕЛИ РУКИ =======================
UPPER_ARM_LENGTH = 3.0
UPPER_ARM_RADIUS = 0.4
UPPER_ARM_AXIS = "y"
UPPER_ARM_SIGN = -1

FOREARM_LENGTH = 2.6
FOREARM_RADIUS = 0.32
FOREARM_AXIS = "y"
FOREARM_SIGN = -1

# Ось для отрисовки сгибания (нутации) в НЕповёрнутой плоскости - см.
# compute_plane_orientation_quat() ниже, которая поворачивает эту ось на
# угол прецессии перед фактическим использованием.
RENDER_AXIS = (1.0, 0.0, 0.0)

# BASIS_FIX_QUAT: замена базиса, найденная и подтверждённая пользователем.
# M = [[0,0,1],[0,-1,0],[1,0,0]] (e_x->(0,0,1), e_y->(0,-1,0), e_z->(1,0,0)) -
# поворот на 180° вокруг оси (1,0,1)/sqrt(2) (det=+1, проверено численно).
# Относится ТОЛЬКО к тому, как выводятся значения наклона/поворота -
# встраивается композицией в R_WORLD_REMAP (см. start_stop_capture_chest_lean()
# ниже), тем же приёмом, что раньше применялись поправки на 180° - НЕ
# применяется ко всей сцене целиком (это была ошибка в предыдущей версии).
_S = 0.7071067811865476  # 1/sqrt(2)
BASIS_FIX_QUAT = (0.0, _S, 0.0, _S)

# BASIS_FIX_QUAT2: ДОПОЛНИТЕЛЬНАЯ поправка (после BASIS_FIX_QUAT оба наклона
# - и вперёд/назад, и влево/вправо - оказались перевёрнуты, а поворот
# остался верным). M2 = diag(-1,1,-1) - поворот на 180° вокруг оси Y (той
# самой, что уже верна для поворота) - разворачивает X и Z, не трогая Y.
# Композируется с BASIS_FIX_QUAT в том же месте (start_stop_capture_chest_lean).
BASIS_FIX_QUAT2 = (0.0, 0.0, 1.0, 0.0)

# ======================= НАСТРОЙКИ МОДЕЛИ ТЕЛА (трапеция) =======================
# Трапециевидная призма: шире сверху (плечи), уже снизу (талия).
# Верх трапеции - на высоте Z=0 (та же высота, откуда "растёт" рука).
TORSO_TOP_HALF_WIDTH = 2.0     # половина ширины на уровне плеч
TORSO_BOTTOM_HALF_WIDTH = 1.2  # половина ширины на уровне талии
TORSO_DEPTH = 1.0              # толщина (перёд-зад) тела
TORSO_HEIGHT = 3.5             # высота трапеции (от плеч до талии)
TORSO_COLOR = (0.35, 0.4, 0.55)

# Точка крепления руки - правое плечо. Определяется опытным путём (визуально
# на экране), сторону поменять - просто сменить знак ниже.
SHOULDER_ATTACH_POINT = (TORSO_TOP_HALF_WIDTH, 0.0, 0.0)

# Локальная ось датчика груди, СОГЛАСНО НАПЕЧАТАННЫМ НА ПЛАТЕ осям (по вашему
# описанию: X=вправо, Y=вниз, Z=вперёд) - "вперёд, наружу от тела". ВАЖНО: по
# факту измерений эта ось оказалась ПОЧТИ ПАРАЛЛЕЛЬНА реально измеренному
# "вниз" (~14° - крепление на практике не соответствует заявленному), поэтому
# build_chest_mount_calibration() ЭТУ константу больше НЕ использует для
# расчёта - там ось "вперёд" теперь выбирается автоматически (см. функцию).
# Используется ТОЛЬКО для чисто информационного compass_heading_deg() ниже -
# на расчёт углов руки/тела не влияет вообще.
CHEST_LOCAL_FORWARD_AXIS = (0.0, 0.0, 1.0)
# ==================================================================

quat_lock = threading.Lock()
current_upper_quat = (1.0, 0.0, 0.0, 0.0)
current_fore_quat = (1.0, 0.0, 0.0, 0.0)
current_chest_quat = (1.0, 0.0, 0.0, 0.0)
running = True

# Режим отображения: BODY - сцена в СК тела (тело неподвижно на экране,
# рука двигается относительно него - как сейчас); WORLD - сцена в СК земли
# (тело физически поворачивается на экране вместе с реальным поворотом
# человека, взятым из датчика груди). Переключается клавишей T.
DISPLAY_MODE_BODY, DISPLAY_MODE_WORLD = range(2)
display_mode = DISPLAY_MODE_BODY

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

# --- калибровка 1 "рука вниз": продольная ось кости В КООРДИНАТАХ ДАТЧИКА,
#     ПЛЮС ориентация датчика груди (нужна только для нескольких
#     диагностических печатей ниже - реальный поворот сцены строится через
#     R_WORLD_REMAP, см. ниже, он фиксированный и калибровки не требует) ---
cal_lock = threading.Lock()
local_bone_axis_upper = None
local_bone_axis_fore = None
is_calibrated_down = False

# --- калибровка 2 "рука вперёд": группа (плечо+предплечье) привязывается
#     к датчику груди по азимуту ---
cal2_lock = threading.Lock()
y_group_in_upper_frame = None   # "вперёд" для группы, в СК датчика плеча
x_group_in_upper_frame = None   # "влево" для группы, в СК датчика плеча
q_chest_cal = None              # показания груди в момент калибровки 2
is_calibrated_forward = False

# --- калибровка 3 "наклон влево" (клавиша L) ---
# ПОЧЕМУ она нужна: датчик на груди физически приклеен с каким-то поворотом
# ВОКРУГ ВЕРТИКАЛИ относительно истинного 'вперёд' тела - неизвестным заранее
# углом (не обязательно 0°/90°/180°). Автоматический выбор 'какая из сырых
# осей X/Y дальше от вниз' ЭТОТ угол в принципе не может учесть - он берёт
# ЧИСТУЮ (1,0,0) или (0,1,0), что верно только если датчик приклеен ИДЕАЛЬНО
# ровно. Отсюда и 'оси под углом, а не перпендикулярно' - никакая перестановка
# знаков (±180°) не может исправить ошибку в ПРОИЗВОЛЬНЫЙ угол, только сама
# ось (не полярность) была неверна. Решение - измерить эту ось РЕАЛЬНЫМ
# движением (тем же приёмом, что раньше калибровали ось плеча взмахом руки).
cal3_lock = threading.Lock()
chest_pitch_axis_measured = None  # измеренная ось 'наклон вперёд', в W_bno (мировая конвенция BNO)
is_world_remap_calibrated = False
R_WORLD_REMAP = IDENTITY_QUAT      # пересчитывается после калибровки L (см. build_world_remap ниже)

CAPTURE_NONE, CAPTURE_DOWN, CAPTURE_CHEST_LEAN = range(3)
capture_mode = CAPTURE_NONE
buf_upper = []
buf_fore = []
buf_chest = []
buf_chest_lean = []


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


def matrix3_to_quat(m00, m01, m02, m10, m11, m12, m20, m21, m22):
    """3x3-матрица (m_ij, i=строка, j=столбец) -> кватернион. Смысл матрицы:
    её СТОЛБЦЫ - это образы стандартных осей (1,0,0),(0,1,0),(0,0,1) после
    поворота, т.е. столбец 0 = m00,m10,m20 = куда переходит (1,0,0)."""
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


# --------------------- Калибровка 1 "рука вниз" + нутация ---------------------

DOWN_WORLD = (0.0, -1.0, 0.0)
UP_WORLD = (0.0, 1.0, 0.0)

# У РАЗНЫХ датчиков РАЗНАЯ конвенция "вертикали" ВНУТРИ их собственного
# кватерниона (не путать с тем, как физически приклеена плата - это другое!).
# У DMP MPU6050 (плечо, предплечье) - Y-конвенция, и AXIS_REMAP уже её
# компенсирует, поэтому для них подходит DOWN_WORLD выше. У BNO08x (грудь) -
# по факту проверки (см. историю калибровки) - Z-конвенция: то, что chip
# сам считает 'вертикалью' внутри своего кватерниона, соответствует оси Z, а
# не Y. Если снова окажется не так - поменяйте эту константу (варианты:
# (0,-1,0), (0,0,-1), (0,0,1) и т.п.) и посмотрите на печатаемый после G
# угол до X/Y/Z - должен быть маленьким (<20°) хотя бы до одной из осей.
CHEST_WORLD_DOWN = (0.0, 0.0, 1.0)


def gravity_in_sensor_frame(q_sensor_to_world, world_down=DOWN_WORLD):
    """Направление 'вниз' В МИРОВОЙ КОНВЕНЦИИ ДАТЧИКА (world_down - см. выше,
    может отличаться у разных датчиков!), выраженное В КООРДИНАТАХ ДАТЧИКА -
    для руки это продольная ось кости, для груди - см. CHEST_WORLD_DOWN."""
    q_world_to_sensor = quat_conjugate(q_sensor_to_world)
    return rotate_vector_by_quat(q_world_to_sensor, world_down)


def axis_from_total_rotation_world(quat_sequence):
    """Ось ОДНОГО суммарного поворота (начало -> конец записи), В МИРОВОЙ
    ('W_bno') конвенции - та же формула (q2 ⊗ conj(q1)), что и в
    get_body_delta_quat(). Знак здесь ОДНОЗНАЧЕН (определяется направлением
    реального физического движения, а не PCA/автовыбором) - тот же приём,
    что раньше решил проблему 'зеркальной' оси при калибровке плеча."""
    if len(quat_sequence) < 2:
        return None
    q1 = quat_sequence[0]
    q2 = quat_sequence[-1]
    dq = quat_multiply(q2, quat_conjugate(q1))
    w, x, y, z = dq
    if w < 0:
        x, y, z = -x, -y, -z
    norm = math.sqrt(x * x + y * y + z * z)
    if norm < 1e-4:
        return None
    return (x / norm, y / norm, z / norm)


def build_world_remap(down_axis, pitch_axis):
    """Строит R_WORLD_REMAP из ДВУХ ИЗМЕРЕННЫХ (не угаданных!) осей, обе - в
    собственной 'мировой' конвенции BNO (W_bno):
      down_axis   - CHEST_WORLD_DOWN (см. выше, свойство самого чипа)
      pitch_axis  - ИЗМЕРЕН калибровкой 'наклон вперёд' (L), см.
                    start_stop_capture_chest_lean() ниже - учитывает
                    РЕАЛЬНЫЙ угол крепления датчика вокруг вертикали, а не
                    чистую (1,0,0) или (0,1,0), как раньше.

    Физика: наклон ВПЕРЁД - это вращение вокруг горизонтальной ОСИ
    ВЛЕВО-ВПРАВО (та же логика, что у самолёта: тангаж - вращение вокруг
    поперечной оси). То есть измеренная этим движением ось - это и есть
    ось 'влево-вправо' сама по себе, напрямую (доворачивать через
    произведение не нужно для НЕЁ - только для 'вперёд', см. ниже).

    'Вперёд' довычисляется через векторное произведение (для право-
    ориентированной, корректной тройки осей)."""
    d = v_norm(down_axis)
    proj = v_dot(d, pitch_axis)
    f = v_norm(v_sub(pitch_axis, v_scale(d, proj)))  # ортогонализация относительно 'вниз'
    l = v_cross(d, f)  # left = down x forward (право-ориентированно для L=(-1,0,0),D=(0,-1,0),F=(0,0,1))

    L_scene = (-1.0, 0.0, 0.0)   # влево в сцене (SHOULDER_ATTACH_POINT = +X = вправо)
    D_scene = DOWN_WORLD
    F_scene = v_cross(L_scene, D_scene)

    q_measured = matrix3_to_quat(l[0], d[0], f[0], l[1], d[1], f[1], l[2], d[2], f[2])
    q_scene = matrix3_to_quat(L_scene[0], D_scene[0], F_scene[0],
                               L_scene[1], D_scene[1], F_scene[1],
                               L_scene[2], D_scene[2], F_scene[2])
    return quat_multiply(q_scene, quat_conjugate(q_measured))


def start_stop_capture_chest_lean():
    """Калибровка 3 (клавиша L): 'наклонитесь корпусом (поклонитесь) вперёд
    ОДИН РАЗ' (не туда-сюда - см. пояснение про однонаправленность движений
    в докстринге калибровки руки выше, та же логика: колебательное движение
    даёт неоднозначный знак оси, одно чистое движение - однозначный).
    Наклон вперёд выбран вместо наклона вбок, потому что его проще сделать
    на больший, более уверенный угол - для калибровки это не принципиально,
    работает любое горизонтальное движение (вторая ось всё равно
    довычисляется через векторное произведение)."""
    global capture_mode, buf_chest_lean
    if capture_mode == CAPTURE_NONE:
        capture_mode = CAPTURE_CHEST_LEAN
        buf_chest_lean = []
        print("[наклон вперёд] запись начата — стоя прямо, наклонитесь корпусом "
              "вперёд ОДИН РАЗ (не туда-сюда), затем L ещё раз - стоп...")
        return
    if capture_mode != CAPTURE_CHEST_LEAN:
        print("Сначала завершите текущую запись.")
        return
    capture_mode = CAPTURE_NONE

    global chest_pitch_axis_measured, is_world_remap_calibrated, R_WORLD_REMAP
    axis = axis_from_total_rotation_world(buf_chest_lean)
    if axis is None:
        print("Слишком мало движения зафиксировано, повторите увереннее.")
        return
    with cal3_lock:
        chest_pitch_axis_measured = axis
        # BASIS_FIX_QUAT / BASIS_FIX_QUAT2 - поправки базиса (найдены и
        # подтверждены пользователем, см. комментарии у констант выше) -
        # относятся ТОЛЬКО к тому, как выводятся значения наклона/поворота
        # (т.е. именно к R_WORLD_REMAP), а не ко всей сцене целиком - тот же
        # приём (композиция поворотом), что раньше применялся вручную
        # (180° вокруг X, затем вокруг Z), теперь встроен туда же.
        basis_fix = quat_multiply(BASIS_FIX_QUAT2, BASIS_FIX_QUAT)
        R_WORLD_REMAP = quat_multiply(basis_fix, build_world_remap(CHEST_WORLD_DOWN, axis))
        is_world_remap_calibrated = True
    print(f"[L] Ось 'влево-вправо' измерена: {tuple(round(c, 3) for c in axis)}. "
          f"Режим 'земля' готов к использованию.")



def start_stop_capture_down():
    global capture_mode, buf_upper, buf_fore, buf_chest
    if capture_mode == CAPTURE_NONE:
        capture_mode = CAPTURE_DOWN
        buf_upper = []
        buf_fore = []
        buf_chest = []
        print("[рука вниз] запись начата — встаньте прямо, рука вдоль тела, "
              "локоть прямой, постойте неподвижно ~2-3 сек...")
        return
    capture_mode = CAPTURE_NONE
    global local_bone_axis_upper, local_bone_axis_fore, is_calibrated_down
    if len(buf_upper) < 5 or len(buf_fore) < 5 or len(buf_chest) < 5:
        print("Слишком мало сэмплов, повторите.")
        return
    g_up = np.mean([gravity_in_sensor_frame(q) for q in buf_upper], axis=0)
    g_fo = np.mean([gravity_in_sensor_frame(q) for q in buf_fore], axis=0)
    g_chest = np.mean([gravity_in_sensor_frame(q, world_down=CHEST_WORLD_DOWN) for q in buf_chest], axis=0)
    with cal_lock:
        local_bone_axis_upper = v_norm(tuple(g_up))
        local_bone_axis_fore = v_norm(tuple(g_fo))
        is_calibrated_down = True
    print(f"[G] Калибровка 'рука вниз' готова. Ось плеча: "
          f"{tuple(round(c, 3) for c in local_bone_axis_upper)}, "
          f"ось предплечья: {tuple(round(c, 3) for c in local_bone_axis_fore)}")
    chest_down_axis = v_norm(tuple(g_chest))
    print(f"[G] Ось 'вниз' датчика груди (в его собственных координатах): "
          f"{tuple(round(c, 3) for c in chest_down_axis)}")
    for name, axis in [("X", (1.0, 0.0, 0.0)), ("Y", (0.0, 1.0, 0.0)), ("Z", (0.0, 0.0, 1.0))]:
        angle = math.degrees(math.acos(max(-1.0, min(1.0, v_dot(chest_down_axis, axis)))))
        print(f"       угол до сырой оси {name}: {angle:.1f}°"
              + ("  <- ближайшая" if angle < 45 else ""))


def nutation_angle_deg(q, local_bone_axis):
    """Угол между текущим направлением кости и 'вниз', arccos(скалярное
    произведение) - БЕЗ проекции на плоскость (см. объяснение в предыдущей
    версии файла: эта формула инвариантна к прецессии по построению)."""
    world_dir = rotate_vector_by_quat(q, local_bone_axis)
    cos_theta = max(-1.0, min(1.0, v_dot(world_dir, DOWN_WORLD)))
    return math.degrees(math.acos(cos_theta))


# --------------------- Калибровка 2 "рука вперёд" (прецессия) ---------------------

def horizontal_project_normalize(v):
    """Убрать вертикальную (Y) компоненту вектора и нормализовать остаток -
    используется, чтобы получить ГОРИЗОНТАЛЬНОЕ направление (азимут)."""
    proj = v_dot(v, UP_WORLD)
    horiz = v_sub(v, v_scale(UP_WORLD, proj))
    return v_norm(horiz)


def try_calibrate_forward():
    """Калибровка 2: 'рука вытянута вперёд' (не важен точный угол в локте
    или горизонт - важен только азимут). Нажимается ОДИН РАЗ, мгновенно
    (не запись, а разовый снимок - в отличие от калибровки 1)."""
    global y_group_in_upper_frame, x_group_in_upper_frame, q_chest_cal, is_calibrated_forward

    with cal_lock:
        if not is_calibrated_down:
            print("Сначала выполните калибровку 1 (G, поза 'рука вниз').")
            return
        bone_axis_upper = local_bone_axis_upper

    with quat_lock:
        q_upper_now = current_upper_quat
        q_chest_now = current_chest_quat

    # направление кости плеча В СОБСТВЕННЫХ координатах датчика плеча,
    # спроецированное на горизонт - ЭТО и есть "вперёд" для всей группы
    bone_dir_own = rotate_vector_by_quat(q_upper_now, bone_axis_upper)
    y_group = horizontal_project_normalize(bone_dir_own)
    if y_group == (0.0, 0.0, 0.0):
        print("Рука сейчас направлена вертикально - нельзя определить азимут, "
              "вытяните руку более горизонтально и повторите.")
        return
    x_group = v_norm(v_cross(UP_WORLD, y_group))  # "влево" - перпендикулярно, право-ориентированный базис

    with cal2_lock:
        y_group_in_upper_frame = y_group
        x_group_in_upper_frame = x_group
        q_chest_cal = q_chest_now
        is_calibrated_forward = True

    print(f"[F] Калибровка 'рука вперёд' готова. Y_group (в СК плеча): "
          f"{tuple(round(c, 3) for c in y_group)}")


def reset_calibration_forward():
    global is_calibrated_forward
    with cal2_lock:
        is_calibrated_forward = False
    print("Калибровка 'рука вперёд' сброшена.")


def twist_angle_about_axis_deg(q, axis):
    """Знаковый угол поворота кватерниона q вокруг ПРОИЗВОЛЬНОЙ оси axis.
    Для датчика груди это ДОЛЖНА быть CHEST_WORLD_DOWN (его собственная
    'мировая' конвенция вертикали, см. выше) - НЕ обязательно (0,1,0)!"""
    w, x, y, z = q
    proj = x * axis[0] + y * axis[1] + z * axis[2]
    return math.degrees(2.0 * math.atan2(proj, w))


def compute_precession_deg():
    """Возвращает (azimuth_arm_deg, body_yaw_since_cal_deg, total_deg) или
    (None, None, None), если калибровка 2 ещё не выполнена.

    azimuth_arm_deg        - насколько РУКА (по её же собственным показаниям,
                              в её ФИКСИРОВАННОЙ/инерциальной СК) отклонилась
                              от направления калибровки. ВКЛЮЧАЕТ в себя и
                              независимое движение руки, И поворот тела (см.
                              докстринг вверху файла) - собственная СК
                              датчика плеча не вращается вместе с телом.
    body_yaw_since_cal_deg - насколько ПОВЕРНУЛОСЬ ТЕЛО с момента калибровки,
                              по датчику груди (магнитометр - без дрейфа).
    total_deg               - РАЗНОСТЬ (не сумма!) - см. докстринг вверху
                              файла: вычитание убирает вклад поворота тела
                              из azimuth_arm, оставляя только движение руки
                              ОТНОСИТЕЛЬНО тела. Используется и для рендера,
                              и как основной результат.
    """
    with cal2_lock:
        if not is_calibrated_forward:
            return None, None, None
        y_group = y_group_in_upper_frame
        x_group = x_group_in_upper_frame
        q_chest_baseline = q_chest_cal

    with cal_lock:
        bone_axis_upper = local_bone_axis_upper

    with quat_lock:
        q_upper_now = current_upper_quat
        q_chest_now = current_chest_quat

    # --- азимут руки, из СОБСТВЕННЫХ показаний датчика плеча ---
    bone_dir_own = rotate_vector_by_quat(q_upper_now, bone_axis_upper)
    horiz_now = horizontal_project_normalize(bone_dir_own)
    cos_a = v_dot(horiz_now, y_group)
    sin_a = v_dot(horiz_now, x_group)
    azimuth_arm_deg = math.degrees(math.atan2(sin_a, cos_a))

    # --- поворот тела с момента калибровки, из датчика груди (стабильно) ---
    # ВАЖНО: ось этого поворота выражена в СОБСТВЕННОЙ 'мировой' конвенции
    # BNO (CHEST_WORLD_DOWN, см. выше) - НЕ в оси Y сцены (это была ошибка,
    # из-за которой в режиме 'тело' поворот тела не убирался из azimuth_arm
    # корректно - в режиме 'земля' её не было видно, там работает другой,
    # уже отдельно исправленный путь через R_WORLD_REMAP).
    q_body_delta = quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))
    body_yaw_since_cal_deg = twist_angle_about_axis_deg(q_body_delta, CHEST_WORLD_DOWN)

    # ВЫЧИТАНИЕ, не сложение - см. докстринг выше и вверху файла
    total_deg = azimuth_arm_deg - body_yaw_since_cal_deg
    return azimuth_arm_deg, body_yaw_since_cal_deg, total_deg


def compute_body_tilt(q_chest, remap_q):
    """
    Возвращает (tilt_forward_deg, tilt_sideways_deg) или (None, None),
    если remap_q не готов.
    tilt_forward > 0  – наклон вперёд,
    tilt_sideways > 0 – наклон вправо (в сторону правого плеча).
    """
    if remap_q is None:
        return None, None
    # Переводим ориентацию груди в систему координат сцены (Y вверх)
    q_scene = quat_multiply(remap_q, q_chest)
    down_vec = rotate_vector_by_quat(q_scene, CHEST_WORLD_DOWN)  # (0,0,1) -> сцена
    # В сцене вертикаль: (0, -1, 0)  (DOWN_WORLD)
    vertical_component = -down_vec[1]          # проекция на вертикаль Y
    forward_component  = -down_vec[2]          # проекция на ось Z (глубина)
    sideways_component =  down_vec[0]          # проекция на ось X
    # pitch – наклон вперёд/назад
    pitch = math.degrees(math.atan2(forward_component, vertical_component))
    # roll – наклон вбок
    roll  = math.degrees(math.atan2(sideways_component, vertical_component))
    return pitch, roll

def get_body_delta_quat():
    """ПОЛНЫЙ (не только yaw-компонента) поворот тела с момента калибровки
    'вперёд', по датчику груди. Используется ТОЛЬКО для режима отображения
    DISPLAY_MODE_WORLD (см. main()) - поворачивает всю сцену (тело+рука) на
    экране вслед за реальным поворотом человека. Возвращает None, если
    калибровка 2 ещё не выполнена."""
    with cal2_lock:
        if not is_calibrated_forward:
            return None
        q_chest_baseline = q_chest_cal
    with quat_lock:
        q_chest_now = current_chest_quat
    return quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))


def compass_heading_deg(q_chest):
    """ЧИСТО информационная величина для консоли - компасный азимут груди
    (0=условный 'север', 90=условный 'восток' и т.д., в зависимости от
    CHEST_LOCAL_FORWARD_AXIS). НЕ используется в расчёте руки/тела выше -
    подберите CHEST_LOCAL_FORWARD_AXIS эмпирически, если нужна точность."""
    world_dir = rotate_vector_by_quat(q_chest, CHEST_LOCAL_FORWARD_AXIS)
    # убираем вертикальную составляющую В КОНВЕНЦИИ САМОГО ДАТЧИКА ГРУДИ
    # (CHEST_WORLD_DOWN, не UP_WORLD сцены - world_dir ещё в конвенции BNO)
    proj = v_dot(world_dir, CHEST_WORLD_DOWN)
    horiz = v_norm(v_sub(world_dir, v_scale(CHEST_WORLD_DOWN, proj)))
    if horiz == (0.0, 0.0, 0.0):
        return None
    # условные "север"/"восток" - любая ортогональная CHEST_WORLD_DOWN пара
    candidates = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    seed = min(candidates, key=lambda a: abs(v_dot(CHEST_WORLD_DOWN, a)))
    north = v_norm(v_sub(seed, v_scale(CHEST_WORLD_DOWN, v_dot(CHEST_WORLD_DOWN, seed))))
    east = v_cross(CHEST_WORLD_DOWN, north)
    cos_a = v_dot(horiz, north)
    sin_a = v_dot(horiz, east)
    heading = math.degrees(math.atan2(sin_a, cos_a))
    return heading % 360.0


# --------------------- Чтение UART ---------------------

def parse_line(line: str):
    try:
        parts = [float(p) for p in line.strip().split(" ")]
        if len(parts) != 12:
            return None
        q_upper = tuple(parts[0:4])
        q_fore = tuple(parts[4:8])
        q_chest = tuple(parts[8:12])
        return q_upper, q_fore, q_chest
    except ValueError:
        return None


def serial_reader_thread():
    global current_upper_quat, current_fore_quat, current_chest_quat, running

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
                q_upper, q_fore, q_chest = parsed
                q_upper = apply_axis_remap(q_upper)
                q_fore = apply_axis_remap(q_fore)
                # ПРИМЕЧАНИЕ: q_chest НЕ пропускаем через AXIS_REMAP датчиков
                # руки - у него своя "мировая" конвенция (CHEST_WORLD_DOWN,
                # см. выше), учитываемая отдельно в нужных местах, а не
                # общим переразмечиванием кватерниона на входе.
                with quat_lock:
                    current_upper_quat = q_upper
                    current_fore_quat = q_fore
                    current_chest_quat = q_chest
                if capture_mode == CAPTURE_DOWN:
                    buf_upper.append(q_upper)
                    buf_fore.append(q_fore)
                    buf_chest.append(q_chest)
                elif capture_mode == CAPTURE_CHEST_LEAN:
                    buf_chest_lean.append(q_chest)
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
        self.target = [0.0, -1.5, 0.0]

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
    y = -5.0
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


def draw_torso():
    """Трапециевидная призма (шире сверху) - грубая модель тела.

    ВАЖНО: в этой сцене вертикаль - ось Y (как и везде в файле: камера,
    DOWN_WORLD, отрисовка руки - все используют Y как 'вверх/вниз').
    Ось Z здесь - глубина (перёд-зад тела), а не высота - НЕ ПУТАТЬ с
    собственной системой координат человека (Y-вперёд, Z-вверх) из докстринга
    вверху файла - та СК используется только для физического смысла осей
    датчиков, а не для координат сцены OpenGL.

    Верх (шире) - на Y=0 (высота плеч, там же откуда 'растёт' рука),
    низ (уже) - на Y=-TORSO_HEIGHT."""
    tw, bw = TORSO_TOP_HALF_WIDTH, TORSO_BOTTOM_HALF_WIDTH
    d = TORSO_DEPTH / 2.0
    y_top, y_bot = 0.0, -TORSO_HEIGHT

    # 8 вершин: верх (шире) и низ (уже). Порядок (x, y, z): x - влево/вправо,
    # y - вверх/вниз (высота), z - перёд/зад (глубина).
    top = [(-tw, y_top, -d), (tw, y_top, -d), (tw, y_top, d), (-tw, y_top, d)]
    bot = [(-bw, y_bot, -d), (bw, y_bot, -d), (bw, y_bot, d), (-bw, y_bot, d)]

    glColor3f(*TORSO_COLOR)

    def quad(p1, p2, p3, p4):
        glBegin(GL_QUADS)
        for p in (p1, p2, p3, p4):
            glVertex3f(*p)
        glEnd()

    quad(*top)                                   # верх
    quad(*bot)                                   # низ
    quad(top[0], top[1], bot[1], bot[0])          # перёд
    quad(top[2], top[3], bot[3], bot[2])          # зад
    quad(top[1], top[2], bot[2], bot[1])          # право
    quad(top[3], top[0], bot[0], bot[3])          # лево


# --------------------- Общий расчёт углов ---------------------

def compute_current_angles():
    """Возвращает (shoulder_deg, elbow_deg) - None, None, если калибровка 1
    ('рука вниз', G) ещё не выполнена. Логика НЕ ИЗМЕНИЛАСЬ по сравнению с
    предыдущей версией файла - прецессия считается отдельно, см.
    compute_precession_deg()."""
    with cal_lock:
        if not is_calibrated_down:
            return None, None
        upper_local_axis = local_bone_axis_upper
        fore_local_axis = local_bone_axis_fore

    with quat_lock:
        q_upper = current_upper_quat
        q_fore = current_fore_quat

    shoulder_deg = nutation_angle_deg(q_upper, upper_local_axis)
    forearm_abs_deg = nutation_angle_deg(q_fore, fore_local_axis)
    elbow_deg = forearm_abs_deg - shoulder_deg

    return shoulder_deg, elbow_deg

def draw_text_overlay(text_lines, display_size):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    w, h = display_size
    gluOrtho2D(0, w, 0, h)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    # Включаем альфа-блендинг
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    font = pygame.font.SysFont("Arial", 16)
    x, y = 10, h - 25
    for line in text_lines:
        text_surf = font.render(line, True, (255, 255, 255))
        text_data = pygame.image.tostring(text_surf, "RGBA", True)
        glRasterPos2i(x, y)
        glDrawPixels(text_surf.get_width(), text_surf.get_height(),
                     GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        y -= text_surf.get_height() + 4

    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()

# --------------------- Основной цикл ---------------------

def main():
    global running

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (1100, 800)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    base_caption = "Nutation+Precession Viewer | G-калибр.вниз F-калибр.вперёд L-калибр.земли R-сброс(F) T-режим 1/2/3-виды 0-камера ESC"
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

    print("Порядок калибровки:")
    print("  1) G — рука вниз, локоть прямой, постоять ~2-3 сек, G ещё раз стоп")
    print("  2) F — рука вытянута вперёд (не важен точный угол/локоть), F -")
    print("        это мгновенный снимок, не запись")
    print("  3) L — наклонитесь корпусом ВПЕРЁД ОДИН РАЗ (не туда-сюда), L ещё")
    print("        раз стоп - нужно для режима 'земля' (T)")
    print("  R — сбросить калибровку 'вперёд' (F)")
    print("  T — переключить режим отображения: СК тела <-> СК земли")

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_t:
                        global display_mode
                        display_mode = (DISPLAY_MODE_WORLD
                                         if display_mode == DISPLAY_MODE_BODY
                                         else DISPLAY_MODE_BODY)
                        mode_name = "ЗЕМЛЯ" if display_mode == DISPLAY_MODE_WORLD else "ТЕЛО"
                        print(f"Режим отображения: {mode_name}")
                    elif event.key == pygame.K_g:
                        start_stop_capture_down()
                    elif event.key == pygame.K_f:
                        try_calibrate_forward()
                    elif event.key == pygame.K_l:
                        start_stop_capture_chest_lean()
                    elif event.key == pygame.K_r:
                        reset_calibration_forward()
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
            azimuth_arm_deg, body_yaw_deg, plane_bearing_deg = compute_precession_deg()

            tilt_fwd = None
            tilt_side = None
            with cal3_lock:
                remap_ready = is_world_remap_calibrated
                remap_q = R_WORLD_REMAP if remap_ready else None
            if remap_ready:
                with quat_lock:
                    q_chest_now = current_chest_quat
                tilt_fwd, tilt_side = compute_body_tilt(q_chest_now, remap_q)

            render_shoulder = shoulder_deg if shoulder_deg is not None else 0.0
            render_elbow = elbow_deg if elbow_deg is not None else 0.0
            render_bearing = plane_bearing_deg if plane_bearing_deg is not None else 0.0

            # поворачиваем ОСЬ сгибания (не саму модель) на угол прецессии -
            # см. докстринг вверху файла, "конъюгационный трюк":
            # rotate(axis, Q) эквивалентно Q ⊗ axis_angle(axis,θ) ⊗ conj(Q)
            q_plane = axis_angle_quat((0.0, 1.0, 0.0), render_bearing)
            q_shoulder_local = axis_angle_quat(RENDER_AXIS, render_shoulder)
            q_elbow_local = axis_angle_quat(RENDER_AXIS, render_elbow)
            q_upper_render = quat_multiply(quat_multiply(q_plane, q_shoulder_local),
                                            quat_conjugate(q_plane))
            q_joint_render = quat_multiply(quat_multiply(q_plane, q_elbow_local),
                                            quat_conjugate(q_plane))

            upper_matrix = quat_to_matrix(q_upper_render)
            joint_matrix = quat_to_matrix(q_joint_render)

            # В режиме "земля" поворачиваем ВСЮ сцену (тело + руку) на
            # реальный поворот тела с момента калибровки. body_delta - это
            # поворот ТЕЛА в СОБСТВЕННОЙ 'мировой' КОНВЕНЦИИ КВАТЕРНИОНА BNO
            # (CHEST_WORLD_DOWN), не в локальных/корпусных осях датчика -
            # поэтому нужен именно R_WORLD_REMAP, теперь ИЗМЕРЕННЫЙ калибровкой
            # 'наклон влево' (L, см. start_stop_capture_chest_lean() выше),
            # а не угаданный автоматически.
            q_world_extra = IDENTITY_QUAT
            if display_mode == DISPLAY_MODE_WORLD:
                with cal3_lock:
                    remap_ready = is_world_remap_calibrated
                    remap_q = R_WORLD_REMAP
                body_delta_raw = get_body_delta_quat()
                if body_delta_raw is not None and remap_ready:
                    q_world_extra = quat_multiply(quat_multiply(remap_q, body_delta_raw),
                                                   quat_conjugate(remap_q))
            world_matrix = quat_to_matrix(q_world_extra)

            glPushMatrix()
            glMultMatrixf(world_matrix)

            draw_torso()

            glPushMatrix()
            glTranslatef(*SHOULDER_ATTACH_POINT)
            glMultMatrixf(upper_matrix)
            draw_segment(UPPER_ARM_LENGTH, UPPER_ARM_RADIUS, UPPER_ARM_AXIS, UPPER_ARM_SIGN)
            tip = segment_tip_offset(UPPER_ARM_LENGTH, UPPER_ARM_AXIS, UPPER_ARM_SIGN)
            glTranslatef(*tip)
            glMultMatrixf(joint_matrix)
            draw_segment(FOREARM_LENGTH, FOREARM_RADIUS, FOREARM_AXIS, FOREARM_SIGN)
            glPopMatrix()

            glPopMatrix()

            text_lines = []
            if shoulder_deg is not None:
                text_lines.append(f"Плечо: {shoulder_deg:6.1f}°")
                text_lines.append(f"Локоть: {elbow_deg:6.1f}°")
            if plane_bearing_deg is not None:
                text_lines.append(f"Прецессия плоскости: {plane_bearing_deg:6.1f}°")
            if tilt_fwd is not None:
                text_lines.append(f"Наклон тела вперёд: {tilt_fwd:.1f}°")
                text_lines.append(f"Наклон тела вбок:   {tilt_side:.1f}°")
            if text_lines:
                draw_text_overlay(text_lines, display)

            pygame.display.flip()

            caption_timer += clock.get_time() / 1000.0
            if caption_timer > 0.2:
                caption_timer = 0.0
                mode_str = "ЗЕМЛЯ" if display_mode == DISPLAY_MODE_WORLD else "ТЕЛО"
                with cal3_lock:
                    remap_ready_now = is_world_remap_calibrated
                world_suffix = ""
                if display_mode == DISPLAY_MODE_WORLD and not remap_ready_now:
                    world_suffix = " [нажмите L для калибровки поворота сцены]"
                if shoulder_deg is not None and plane_bearing_deg is not None:
                    with quat_lock:
                        q_chest_now = current_chest_quat
                    heading = compass_heading_deg(q_chest_now)
                    heading_str = f"{heading:5.1f}°" if heading is not None else "?"
                    pygame.display.set_caption(
                        f"{base_caption} | [{mode_str}] плечо:{shoulder_deg:6.1f} локоть:{elbow_deg:6.1f} "
                        f"прецессия:{plane_bearing_deg:6.1f} (рука:{azimuth_arm_deg:5.1f} "
                        f"тело:{body_yaw_deg:5.1f}) компас_тела:{heading_str}{world_suffix}"
                    )
                elif shoulder_deg is None:
                    pygame.display.set_caption(f"{base_caption} | [{mode_str}] НЕ ОТКАЛИБРОВАНО (нажмите G){world_suffix}")
                else:
                    pygame.display.set_caption(f"{base_caption} | [{mode_str}] прецессия не откалибр. (нажмите F){world_suffix}")

            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()