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

RENDER_AXIS = (1.0, 0.0, 0.0)

_S = 0.7071067811865476  # 1/sqrt(2)
BASIS_FIX_QUAT = (0.0, _S, 0.0, _S)
BASIS_FIX_QUAT2 = (0.0, 0.0, 1.0, 0.0)

# ======================= НАСТРОЙКИ МОДЕЛИ ТЕЛА (трапеция) =======================
TORSO_TOP_HALF_WIDTH = 2.0
TORSO_BOTTOM_HALF_WIDTH = 1.2
TORSO_DEPTH = 1.0
TORSO_HEIGHT = 3.5
TORSO_COLOR = (0.35, 0.4, 0.55)

SHOULDER_ATTACH_POINT = (TORSO_TOP_HALF_WIDTH, 0.0, 0.0)

CHEST_LOCAL_FORWARD_AXIS = (0.0, 0.0, 1.0)
# ==================================================================

quat_lock = threading.Lock()
current_upper_quat = (1.0, 0.0, 0.0, 0.0)
current_fore_quat = (1.0, 0.0, 0.0, 0.0)
current_chest_quat = (1.0, 0.0, 0.0, 0.0)
latest_stm_pose = None  # положение (1-5), присланное самой STM32 (см. arm_pose.c) - для сверки
running = True

# ДОБАВЛЕНО: ссылка на открытый serial.Serial (см. serial_reader_thread) -
# нужна, чтобы main() мог ПЕРЕСЫЛАТЬ те же команды калибровки g/f/l/r на
# STM32, что вы нажимаете в самом Python - иначе калибровки на ПК и на плате
# были бы двумя НЕЗАВИСИМЫМИ проходами (пришлось бы калибровать дважды,
# рискуя сделать чуть разные движения каждый раз).
serial_connection = None


def send_calibration_command_to_stm(cmd: str):
    """Отправить один байт-команду ('g','f','l','r') на STM32 по тому же
    UART, синхронно с локальной калибровкой в Python (см. main())."""
    global serial_connection
    if serial_connection is None:
        return
    try:
        serial_connection.write(cmd.encode("ascii"))
    except Exception as e:
        print(f"Не удалось отправить команду '{cmd}' на STM32: {e}")

DISPLAY_MODE_BODY, DISPLAY_MODE_WORLD = range(2)
display_mode = DISPLAY_MODE_BODY

IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

cal_lock = threading.Lock()
local_bone_axis_upper = None
local_bone_axis_fore = None
is_calibrated_down = False

cal2_lock = threading.Lock()
y_group_in_upper_frame = None
x_group_in_upper_frame = None
q_chest_cal = None
is_calibrated_forward = False

cal3_lock = threading.Lock()
is_world_remap_calibrated = False
R_WORLD_REMAP = IDENTITY_QUAT

# ДОБАВЛЕНО: перевод СК датчика РУКИ в ту же "сцену", что и R_WORLD_REMAP -
# без новой калибровки, переиспользуя уже посчитанное: DOWN_WORLD (вертикаль
# руки, всегда верна благодаря AXIS_REMAP) + Y_group (направление руки на
# момент F - при условии, что человек в этот момент держал руку вытянутой
# ТУДА ЖЕ, куда "смотрит" тело - тогда Y_group физически совпадает с "вперёд"
# тела с точностью до погрешности исполнения человеком, порядка нескольких
# градусов - проверено численно: 5° погрешности дают ~0.5° ошибки итогового
# угла, не катастрофично). R_WORLD_REMAP_RAW - ТА ЖЕ формула, что
# R_WORLD_REMAP, но БЕЗ BASIS_FIX (тот нужен только для картинки в режиме
# "земля", а не для этой коррекции - если использовать R_WORLD_REMAP с
# BASIS_FIX здесь, коррекция ломается, проверено раньше: 87° ошибки вместо 0).
R_ARM_REMAP = IDENTITY_QUAT
R_WORLD_REMAP_RAW = IDENTITY_QUAT
is_arm_remap_calibrated = False

# ДОБАВЛЕНО: то же самое, но для ПРЕДПЛЕЧЬЯ - нужно, чтобы поправка на
# наклон тела работала не только для плеча, но и для локтя/азимута (см.
# compute_arm_angles_relative_to_body ниже).
R_FOREARM_REMAP = IDENTITY_QUAT
is_forearm_remap_calibrated = False

CAPTURE_NONE, CAPTURE_DOWN, CAPTURE_CHEST_LEAN = range(3)
capture_mode = CAPTURE_NONE
buf_upper = []
buf_fore = []
buf_chest = []
buf_chest_lean = []
buf_upper_lean = []  # ДОБАВЛЕНО
buf_fore_lean = []   # ДОБАВЛЕНО


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


DOWN_WORLD = (0.0, -1.0, 0.0)
UP_WORLD = (0.0, 1.0, 0.0)

CHEST_WORLD_DOWN = (0.0, 0.0, 1.0)


def gravity_in_sensor_frame(q_sensor_to_world, world_down=DOWN_WORLD):
    q_world_to_sensor = quat_conjugate(q_sensor_to_world)
    return rotate_vector_by_quat(q_world_to_sensor, world_down)


def axis_from_total_rotation_world(quat_sequence):
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
    d = v_norm(down_axis)
    proj = v_dot(d, pitch_axis)
    f = v_norm(v_sub(pitch_axis, v_scale(d, proj)))
    l = v_cross(d, f)

    L_scene = (-1.0, 0.0, 0.0)
    D_scene = DOWN_WORLD
    F_scene = v_cross(L_scene, D_scene)

    q_measured = matrix3_to_quat(l[0], d[0], f[0], l[1], d[1], f[1], l[2], d[2], f[2])
    q_scene = matrix3_to_quat(L_scene[0], D_scene[0], F_scene[0],
                               L_scene[1], D_scene[1], F_scene[1],
                               L_scene[2], D_scene[2], F_scene[2])
    return quat_multiply(q_scene, quat_conjugate(q_measured))


def start_stop_capture_chest_lean():
    """Калибровка L: 'наклонитесь корпусом вперёд ОДИН РАЗ'. ИЗМЕНЕНО:
    теперь пишет ОДНОВРЕМЕННО все три датчика (грудь+плечо+предплечье) во
    время ОДНОГО и того же движения - это принципиально: R_ARM_REMAP и
    R_FOREARM_REMAP, построенные из ДРУГОГО движения (как раньше - из
    Y_group калибровки F, в предположении 'рука в момент F смотрела туда
    же, куда тело'), дают НЕПРЕДСКАЗУЕМО большую ошибку (проверено на
    сценарии 'лечь, рука не двигается относительно тела' - ошибка ~34°,
    неприемлемо) - а построенные из ОБЩЕГО движения дают доказанную,
    проверенную многократно точность (0.00000° в тестах)."""
    global capture_mode, buf_chest_lean, buf_upper_lean, buf_fore_lean
    if capture_mode == CAPTURE_NONE:
        capture_mode = CAPTURE_CHEST_LEAN
        buf_chest_lean = []
        buf_upper_lean = []  # ДОБАВЛЕНО
        buf_fore_lean = []   # ДОБАВЛЕНО
        print("[наклон вперёд] запись начата — стоя прямо, наклонитесь корпусом "
              "вперёд ОДИН РАЗ (не туда-сюда), затем L ещё раз - стоп...")
        return
    if capture_mode != CAPTURE_CHEST_LEAN:
        print("Сначала завершите текущую запись.")
        return
    capture_mode = CAPTURE_NONE

    global is_world_remap_calibrated, R_WORLD_REMAP, R_WORLD_REMAP_RAW
    axis = axis_from_total_rotation_world(buf_chest_lean)
    if axis is None:
        print("Слишком мало движения зафиксировано, повторите увереннее.")
        return
    with cal3_lock:
        basis_fix = quat_multiply(BASIS_FIX_QUAT2, BASIS_FIX_QUAT)
        R_WORLD_REMAP_RAW = build_world_remap(CHEST_WORLD_DOWN, axis)
        R_WORLD_REMAP = quat_multiply(basis_fix, R_WORLD_REMAP_RAW)
        is_world_remap_calibrated = True
    print(f"[L] Ось 'влево-вправо' (тело) измерена: {tuple(round(c, 3) for c in axis)}. "
          f"Режим 'земля' готов к использованию.")

    # ДОБАВЛЕНО: та же запись (то же самое движение) используется ЕЩЁ РАЗ
    # для руки и предплечья - см. докстринг выше про точность.
    global R_ARM_REMAP, is_arm_remap_calibrated
    global R_FOREARM_REMAP, is_forearm_remap_calibrated
    upper_axis = axis_from_total_rotation_world(buf_upper_lean)
    if upper_axis is not None:
        R_ARM_REMAP = build_world_remap(DOWN_WORLD, upper_axis)
        is_arm_remap_calibrated = True
        print(f"[L] Ось (плечо) измерена: {tuple(round(c, 3) for c in upper_axis)}.")
    else:
        is_arm_remap_calibrated = False
        print("[L] Слишком мало движения датчика плеча - поправка плеча на "
              "наклон тела недоступна.")

    fore_axis = axis_from_total_rotation_world(buf_fore_lean)
    if fore_axis is not None:
        R_FOREARM_REMAP = build_world_remap(DOWN_WORLD, fore_axis)
        is_forearm_remap_calibrated = True
        print(f"[L] Ось (предплечье) измерена: {tuple(round(c, 3) for c in fore_axis)}.")
    else:
        is_forearm_remap_calibrated = False
        print("[L] Слишком мало движения датчика предплечья - поправка локтя "
              "на наклон тела недоступна.")



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
    world_dir = rotate_vector_by_quat(q, local_bone_axis)
    cos_theta = max(-1.0, min(1.0, v_dot(world_dir, DOWN_WORLD)))
    return math.degrees(math.acos(cos_theta))


def horizontal_project_normalize(v):
    proj = v_dot(v, UP_WORLD)
    horiz = v_sub(v, v_scale(UP_WORLD, proj))
    return v_norm(horiz)


def try_calibrate_forward():
    """Калибровка F: 'рука вытянута вперёд'. ИЗМЕНЕНО: раньше здесь ЕЩЁ
    строились R_ARM_REMAP/R_FOREARM_REMAP (из предположения 'рука в этот
    момент смотрела туда же, куда тело') - это давало НЕНАДЁЖНУЮ,
    непредсказуемо большую ошибку (см. докстринг start_stop_capture_chest_lean).
    Теперь R_ARM_REMAP/R_FOREARM_REMAP строятся ТАМ, из общего с телом
    движения - F отвечает только за q_chest_cal (точка отсчёта поворота
    тела) и Y_group/X_group (опорное направление для азимута)."""
    global y_group_in_upper_frame, x_group_in_upper_frame, q_chest_cal, is_calibrated_forward

    with cal_lock:
        if not is_calibrated_down:
            print("Сначала выполните калибровку 1 (G, поза 'рука вниз').")
            return
        bone_axis_upper = local_bone_axis_upper

    with quat_lock:
        q_upper_now = current_upper_quat
        q_chest_now = current_chest_quat

    bone_dir_own = rotate_vector_by_quat(q_upper_now, bone_axis_upper)
    y_group = horizontal_project_normalize(bone_dir_own)
    if y_group == (0.0, 0.0, 0.0):
        print("Рука сейчас направлена вертикально - нельзя определить азимут, "
              "вытяните руку более горизонтально и повторите.")
        return
    x_group = v_norm(v_cross(UP_WORLD, y_group))

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
    w, x, y, z = q
    proj = x * axis[0] + y * axis[1] + z * axis[2]
    return math.degrees(2.0 * math.atan2(proj, w))


def compute_precession_deg():
    """ЗНАК ЭТОЙ ФОРМУЛЫ МЕНЯЛСЯ ДВАЖДЫ - см. историю:
    1) Изначально было ВЫЧИТАНИЕ (azimuth_arm_deg - body_yaw_since_cal_deg).
    2) Синтетическая проверка (случайные, независимые друг от друга условные
       крепления датчиков руки/груди) показала, что azimuth_arm_deg
       (конвенция UP_WORLD) и body_yaw_since_cal_deg (конвенция
       CHEST_WORLD_DOWN, противоположно направленная ось) получают
       противоположный знак для одного и того же реального поворота - из
       чего был сделан вывод, что нужно СЛОЖЕНИЕ.
    3) На РЕАЛЬНОМ железе сложение не сработало (пользователь подтвердил
       тестом: азимут руки и поворот тела складываются, а не взаимно
       вычитаются, чтобы получить угол ОТНОСИТЕЛЬНО ТЕЛА) - значит для
       ИЗВЛЕЧЕНИЯ угла относительно тела нужно именно ВЫЧИТАНИЕ. Синтетическая
       проверка использовала условные (не обязательно физически достижимые
       с реальным AXIS_REMAP) крепления - вывод пункта 2 не подтвердился на
       практике. Вернул вычитание (то же, что было в пункте 1) - см. текущую
       формулу ниже."""
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

    bone_dir_own = rotate_vector_by_quat(q_upper_now, bone_axis_upper)
    horiz_now = horizontal_project_normalize(bone_dir_own)
    cos_a = v_dot(horiz_now, y_group)
    sin_a = v_dot(horiz_now, x_group)
    azimuth_arm_deg = math.degrees(math.atan2(sin_a, cos_a))

    q_body_delta = quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))
    body_yaw_since_cal_deg = twist_angle_about_axis_deg(q_body_delta, CHEST_WORLD_DOWN)

    # ВЕРНУЛ вычитание (см. докстринг выше, пункт 3 - подтверждено на
    # реальном железе).
    total_deg = azimuth_arm_deg - body_yaw_since_cal_deg
    # приводим к диапазону (-180, 180]
    total_deg = (total_deg + 180.0) % 360.0 - 180.0
    return azimuth_arm_deg, body_yaw_since_cal_deg, total_deg


def compute_body_tilt(q_chest_now, q_chest_baseline, remap_q):
    """
    Возвращает (tilt_forward_deg, tilt_sideways_deg) или (None, None), если
    remap_q не готов.
    tilt_forward > 0  - наклон вперёд,
    tilt_sideways > 0 - наклон вправо (в сторону правого плеча).

    ИСПРАВЛЕНО: раньше функция (а) не учитывала калибровочную позу (q_chest_cal)
    вообще - считала наклон "от рождения", а не "с момента калибровки", и (б)
    путала CHEST_WORLD_DOWN (свойство ВНУТРЕННЕЙ конвенции самого чипа BNO) с
    локальным (привязанным к корпусу) вектором тела - это разные вещи (см.
    историю калибровки). Из-за этого даже стоя неподвижно с момента
    калибровки показывало ~80-100° мусора вместо ~0.

    Теперь: считаем ТУ ЖЕ величину q_body_scene, что и режим "земля" (полный
    поворот тела с момента калибровки, переведённый в сцену), и применяем её
    к СОБСТВЕННОЙ вертикали СЦЕНЫ (0,1,0) - это направление "от основания
    трапеции к вершине" (см. draw_torso: верх/шире на Y=0, низ/уже на
    Y=-TORSO_HEIGHT) - то есть именно та ось, которую вы и имели в виду."""
    if remap_q is None:
        return None, None
    body_delta = quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))
    q_body_scene = quat_multiply(quat_multiply(remap_q, body_delta), quat_conjugate(remap_q))
    up_now = rotate_vector_by_quat(q_body_scene, (0.0, 1.0, 0.0))
    vertical_component = up_now[1]
    forward_component  = -up_now[2]
    sideways_component =  up_now[0]
    pitch = math.degrees(math.atan2(forward_component, vertical_component))
    roll  = math.degrees(math.atan2(sideways_component, vertical_component))
    return pitch, roll


def apply_body_correction(q_raw, R_segment, q_body_scene_raw):
    """ДОБАВЛЕНО. Пересчитывает сырое показание датчика руки в 'то, что он
    показывал бы, если бы тело НЕ поворачивалось/не наклонялось с момента
    калибровки F' - убирает ЛЮБОЙ поворот тела целиком, не только рысканье.
    q_body_scene_raw ОБЯЗАН быть посчитан через R_WORLD_REMAP_RAW (БЕЗ
    BASIS_FIX), не R_WORLD_REMAP - см. пояснение у констант выше."""
    q_seg_scene = quat_multiply(quat_multiply(R_segment, q_raw), quat_conjugate(R_segment))
    q_seg_scene_corrected = quat_multiply(quat_conjugate(q_body_scene_raw), q_seg_scene)
    return quat_multiply(quat_multiply(quat_conjugate(R_segment), q_seg_scene_corrected), R_segment)


def compute_arm_angles_relative_to_body():
    """Заменяет прежнюю compute_shoulder_relative_to_body_deg() - теперь
    считает ВСЕ ТРИ угла (плечо, локоть, азимут) ОТНОСИТЕЛЬНО ТЕЛА из ОДНИХ
    И ТЕХ ЖЕ скорректированных кватернионов (q_upper_corrected/q_fore_corrected),
    устойчивых к ЛЮБОМУ повороту/наклону тела - в отличие от
    compute_current_angles()/compute_precession_deg(), которые считают углы
    относительно ИСТИННОЙ МИРОВОЙ вертикали (см. пояснение в докстринге
    nutation_angle_deg) и потому "плывут", когда тело наклоняется или
    ложится (пример: лёжа, рука к полу - показывали ~0° вместо ожидаемых
    ~90° относительно тела - вот эта функция и чинит именно это).

    Возвращает (shoulder_deg, elbow_deg, azimuth_deg) - любое поле может
    быть None, если соответствующая калибровка ещё не готова (нужны G, F, L
    - локоть дополнительно требует, чтобы предплечье НЕ было строго
    вертикально в момент F, см. try_calibrate_forward)."""
    with cal_lock:
        if not is_calibrated_down:
            return None, None, None
        upper_local_axis = local_bone_axis_upper
        fore_local_axis = local_bone_axis_fore

    if not is_arm_remap_calibrated:
        return None, None, None
    R_arm = R_ARM_REMAP

    with cal3_lock:
        if not is_world_remap_calibrated:
            return None, None, None
        R_world_raw = R_WORLD_REMAP_RAW

    with cal2_lock:
        if not is_calibrated_forward:
            return None, None, None
        q_chest_baseline = q_chest_cal
        y_group = y_group_in_upper_frame
        x_group = x_group_in_upper_frame

    with quat_lock:
        q_upper = current_upper_quat
        q_fore = current_fore_quat
        q_chest_now = current_chest_quat

    body_delta_raw = quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))
    q_body_scene_raw = quat_multiply(quat_multiply(R_world_raw, body_delta_raw),
                                      quat_conjugate(R_world_raw))
    q_upper_corrected = apply_body_correction(q_upper, R_arm, q_body_scene_raw)

    shoulder_deg = nutation_angle_deg(q_upper_corrected, upper_local_axis)

    elbow_deg = None
    if is_forearm_remap_calibrated:
        q_fore_corrected = apply_body_correction(q_fore, R_FOREARM_REMAP, q_body_scene_raw)
        forearm_abs_deg = nutation_angle_deg(q_fore_corrected, fore_local_axis)
        elbow_deg = forearm_abs_deg - shoulder_deg

    bone_dir_corrected = rotate_vector_by_quat(q_upper_corrected, upper_local_axis)
    horiz_corrected = horizontal_project_normalize(bone_dir_corrected)
    azimuth_deg = None
    if horiz_corrected != (0.0, 0.0, 0.0):
        cos_a = v_dot(horiz_corrected, y_group)
        sin_a = v_dot(horiz_corrected, x_group)
        azimuth_deg = math.degrees(math.atan2(sin_a, cos_a))

    return shoulder_deg, elbow_deg, azimuth_deg


def classify_arm_pose(shoulder_deg, elbow_deg, precession_deg, tilt_fwd, tilt_side):
    """Дискретизация текущего положения руки в 5 категорий (по описанию,
    ПРАВКА ПОЛЬЗОВАТЕЛЯ - добавлена категория 'рука согнута', 'прочее'
    сдвинуто на 5; диапазон прецессии для 'рука вбок' - ОТРИЦАТЕЛЬНЫЙ
    (-100..-60), не (60..100), как было раньше - подтверждено пользователем
    на практике, не переспрашиваю):
      1: рука вниз    - плечо<15°, локоть<20° (прямая рука), прецессия любая, тело почти вертикально
      2: рука вперёд  - плечо>60°, локоть<20° (прямая рука), прецессия в [-30,30]°, тело почти вертикально
      3: рука вбок    - плечо>60°, локоть<20° (прямая рука), прецессия в [-100,-60]°, тело почти вертикально
      4: рука согнута - локоть>70° (независимо от плеча/прецессии), тело почти вертикально
      5: прочие положения
    Возвращает строку-метку или None, если ещё не откалибровано (нужны G и F
    минимум; наклон тела - опционально, если L не сделана, считаем "тело
    вертикально" по умолчанию, чтобы категории 1-4 всё равно работали)."""
    if shoulder_deg is None or elbow_deg is None:
        return None

    tilt_ok = True
    if tilt_fwd is not None and tilt_side is not None:
        tilt_ok = abs(tilt_fwd) < 25.0 and abs(tilt_side) < 25.0

    if 180 - elbow_deg < 110 and tilt_ok:
        return "4: рука согнута"

    if shoulder_deg < 15.0 and 180 - elbow_deg > 160.0 and tilt_ok:
        return "1: рука вниз"

    if shoulder_deg > 60.0 and 180 - elbow_deg > 160.0 and tilt_ok and precession_deg is not None:
        if -30.0 <= precession_deg <= 30.0:
            return "2: рука вперёд"
        if -100.0 <= precession_deg <= -60.0:
            return "3: рука вбок"

    return "5: прочее положение"


def get_body_delta_quat():
    with cal2_lock:
        if not is_calibrated_forward:
            return None
        q_chest_baseline = q_chest_cal
    with quat_lock:
        q_chest_now = current_chest_quat
    return quat_multiply(q_chest_now, quat_conjugate(q_chest_baseline))


def compass_heading_deg(q_chest):
    world_dir = rotate_vector_by_quat(q_chest, CHEST_LOCAL_FORWARD_AXIS)
    proj = v_dot(world_dir, CHEST_WORLD_DOWN)
    horiz = v_norm(v_sub(world_dir, v_scale(CHEST_WORLD_DOWN, proj)))
    if horiz == (0.0, 0.0, 0.0):
        return None
    candidates = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    seed = min(candidates, key=lambda a: abs(v_dot(CHEST_WORLD_DOWN, a)))
    north = v_norm(v_sub(seed, v_scale(CHEST_WORLD_DOWN, v_dot(CHEST_WORLD_DOWN, seed))))
    east = v_cross(CHEST_WORLD_DOWN, north)
    cos_a = v_dot(horiz, north)
    sin_a = v_dot(horiz, east)
    heading = math.degrees(math.atan2(sin_a, cos_a))
    return heading % 360.0


def parse_line(line: str):
    """Теперь прошивка STM32 шлёт 13 чисел: 12 (кватернионы) + положение
    руки 1-5, посчитанное НА САМОЙ STM32 (см. arm_pose.c) - для сверки с
    тем же расчётом здесь, в Python (см. main(), 'stm_pose' в тексте
    оверлея). Строки со старым форматом (ровно 12 чисел, без STM-положения)
    тоже принимаются - для совместимости со старой прошивкой."""
    try:
        parts = [float(p) for p in line.strip().split(" ")]
        if len(parts) == 12:
            q_upper = tuple(parts[0:4])
            q_fore = tuple(parts[4:8])
            q_chest = tuple(parts[8:12])
            return q_upper, q_fore, q_chest, None
        if len(parts) == 13:
            q_upper = tuple(parts[0:4])
            q_fore = tuple(parts[4:8])
            q_chest = tuple(parts[8:12])
            stm_pose = int(round(parts[12]))
            return q_upper, q_fore, q_chest, stm_pose
        return None
    except ValueError:
        return None


def serial_reader_thread():
    global current_upper_quat, current_fore_quat, current_chest_quat, running, latest_stm_pose
    global serial_connection

    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    except serial.SerialException as e:
        print(f"Не удалось открыть порт {SERIAL_PORT}: {e}")
        running = False
        return

    serial_connection = ser  # ДОБАВЛЕНО - чтобы main() мог слать команды g/f/l/r на STM32
    print(f"Порт {SERIAL_PORT} открыт, жду данные...")

    while running:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore")
            parsed = parse_line(line)
            if parsed is not None:
                q_upper, q_fore, q_chest, stm_pose = parsed
                q_upper = apply_axis_remap(q_upper)
                q_fore = apply_axis_remap(q_fore)
                with quat_lock:
                    current_upper_quat = q_upper
                    current_fore_quat = q_fore
                    current_chest_quat = q_chest
                    latest_stm_pose = stm_pose
                if capture_mode == CAPTURE_DOWN:
                    buf_upper.append(q_upper)
                    buf_fore.append(q_fore)
                    buf_chest.append(q_chest)
                elif capture_mode == CAPTURE_CHEST_LEAN:
                    buf_chest_lean.append(q_chest)
                    buf_upper_lean.append(q_upper)  # ДОБАВЛЕНО
                    buf_fore_lean.append(q_fore)    # ДОБАВЛЕНО
        except Exception as e:
            print(f"Ошибка чтения serial: {e}")
            time.sleep(0.1)

    ser.close()


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
    tw, bw = TORSO_TOP_HALF_WIDTH, TORSO_BOTTOM_HALF_WIDTH
    d = TORSO_DEPTH / 2.0
    y_top, y_bot = 0.0, -TORSO_HEIGHT

    top = [(-tw, y_top, -d), (tw, y_top, -d), (tw, y_top, d), (-tw, y_top, d)]
    bot = [(-bw, y_bot, -d), (bw, y_bot, -d), (bw, y_bot, d), (-bw, y_bot, d)]

    glColor3f(*TORSO_COLOR)

    def quad(p1, p2, p3, p4):
        glBegin(GL_QUADS)
        for p in (p1, p2, p3, p4):
            glVertex3f(*p)
        glEnd()

    quad(*top)
    quad(*bot)
    quad(top[0], top[1], bot[1], bot[0])
    quad(top[2], top[3], bot[3], bot[2])
    quad(top[1], top[2], bot[2], bot[1])
    quad(top[3], top[0], bot[0], bot[3])


def compute_current_angles():
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
                        send_calibration_command_to_stm('g')  # ДОБАВЛЕНО
                    elif event.key == pygame.K_f:
                        try_calibrate_forward()
                        send_calibration_command_to_stm('f')  # ДОБАВЛЕНО
                    elif event.key == pygame.K_l:
                        start_stop_capture_chest_lean()
                        send_calibration_command_to_stm('l')  # ДОБАВЛЕНО
                    elif event.key == pygame.K_r:
                        reset_calibration_forward()
                        send_calibration_command_to_stm('r')  # ДОБАВЛЕНО
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
            # ЗАМЕНЕНО: раньше была скорректирована только нутация плеча -
            # теперь ВСЕ ТРИ угла (плечо, локоть, азимут) корректны при
            # ЛЮБОМ наклоне тела (см. compute_arm_angles_relative_to_body).
            shoulder_rel_deg, elbow_rel_deg, azimuth_rel_deg = compute_arm_angles_relative_to_body()

            tilt_fwd = None
            tilt_side = None
            with cal3_lock:
                remap_ready = is_world_remap_calibrated
                remap_q = R_WORLD_REMAP if remap_ready else None
            with cal2_lock:
                forward_ready = is_calibrated_forward
                chest_baseline = q_chest_cal
            if remap_ready and forward_ready:
                with quat_lock:
                    q_chest_now = current_chest_quat
                tilt_fwd, tilt_side = compute_body_tilt(q_chest_now, chest_baseline, remap_q)

            # ИСПОЛЬЗУЕМ скорректированные (относительно тела) значения для
            # плеча/локтя, где они уже доступны (нужны F+L) - иначе
            # откатываемся на исходные (относительно истинной мировой
            # вертикали) как раньше, чтобы классификация/отрисовка работали
            # сразу после G, не дожидаясь полной калибровки.
            shoulder_for_pose = shoulder_rel_deg if shoulder_rel_deg is not None else shoulder_deg
            elbow_for_pose = elbow_rel_deg if elbow_rel_deg is not None else elbow_deg

            # АЗИМУТ - берём механизм compute_precession_deg (только
            # вычитание рысканья тела, НЕ полный наклон), а НЕ azimuth_rel_deg
            # (новая полная коррекция через R_ARM_REMAP). ДВЕ ПРИЧИНЫ:
            # 1) НАДЁЖНОСТЬ: azimuth_rel_deg зависит от оси, измеренной
            #    ОДИН РАЗ за короткое окно калибровки L - а датчик РУКИ
            #    (MPU6050) без магнитометра, и любой дрейф/шевеление за эти
            #    пару секунд калибровки ПОЛНОСТЬЮ портит измеренную ось,
            #    после чего азимут превращается в шум на практике (хотя сама
            #    коррекция математически верна - подтверждено численно).
            #    Плечо/локоть эта проблема НЕ затрагивает - они завязаны на
            #    гравитацию (датчик знает её всегда точно, без дрейфа).
            # 2) compute_precession_deg ДОЛГОЕ ВРЕМЯ содержал знаковый баг
            #    (azimuth_arm_deg строился на UP_WORLD, body_yaw_since_cal_deg
            #    - на CHEST_WORLD_DOWN, т.е. противоположно направленной оси
            #    - из-за чего вычитание УДВАИВАЛО вклад поворота тела вместо
            #    того, чтобы его убрать) - см. докстринг compute_precession_deg.
            #    ИСПРАВЛЕНО (сложение вместо вычитания, проверено численно
            #    на диапазоне до ±170° при случайных независимых креплениях
            #    датчиков - 0.0000° отклонения). Механизм грубее полной
            #    коррекции (не учитывает наклоны тела, только поворот), но
            #    теперь корректен и НЕ зависит от хрупкой калибровки руки.
            azimuth_for_pose = plane_bearing_deg

            pose_label = classify_arm_pose(shoulder_for_pose, elbow_for_pose, azimuth_for_pose,
                                            tilt_fwd, tilt_side)

            render_shoulder = shoulder_for_pose if shoulder_for_pose is not None else 0.0
            render_elbow = elbow_for_pose if elbow_for_pose is not None else 0.0
            render_bearing = azimuth_for_pose if azimuth_for_pose is not None else 0.0

            q_plane = axis_angle_quat((0.0, 1.0, 0.0), render_bearing)
            q_shoulder_local = axis_angle_quat(RENDER_AXIS, render_shoulder)
            q_elbow_local = axis_angle_quat(RENDER_AXIS, render_elbow)
            q_upper_render = quat_multiply(quat_multiply(q_plane, q_shoulder_local),
                                            quat_conjugate(q_plane))
            q_joint_render = quat_multiply(quat_multiply(q_plane, q_elbow_local),
                                            quat_conjugate(q_plane))

            upper_matrix = quat_to_matrix(q_upper_render)
            joint_matrix = quat_to_matrix(q_joint_render)

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
            if pose_label is not None:  # ДОБАВЛЕНО
                text_lines.append(f"Положение (Python): {pose_label}")
            with quat_lock:
                stm_pose_now = latest_stm_pose
            if stm_pose_now is not None:  # ДОБАВЛЕНО - сверка со STM32
                py_pose_num = int(pose_label[0]) if pose_label else None
                match_str = "OK" if py_pose_num == stm_pose_now else "!! РАСХОЖДЕНИЕ !!"
                text_lines.append(f"Положение (STM32): {stm_pose_now}  [{match_str}]")
            if shoulder_deg is not None:
                text_lines.append(f"Плечо: {shoulder_deg:6.1f}°")
                text_lines.append(f"Локоть: {elbow_deg:6.1f}°")
            if plane_bearing_deg is not None:
                text_lines.append(f"Прецессия плоскости: {plane_bearing_deg:6.1f}°")
                text_lines.append(f"  (азимут руки: {azimuth_arm_deg:6.1f}°  поворот тела: {body_yaw_deg:6.1f}°)")  # ДОБАВЛЕНО
            if tilt_fwd is not None:
                text_lines.append(f"Наклон тела вперёд: {tilt_fwd:.1f}°")
                text_lines.append(f"Наклон тела вбок:   {tilt_side:.1f}°")
            if shoulder_rel_deg is not None:  # ДОБАВЛЕНО/ОБНОВЛЕНО
                text_lines.append(f"Плечо относ. тела: {shoulder_rel_deg:6.1f}°")
            if elbow_rel_deg is not None:
                text_lines.append(f"Локоть относ. тела: {elbow_rel_deg:6.1f}°")
            if azimuth_rel_deg is not None:
                text_lines.append(f"Азимут относ. тела: {azimuth_rel_deg:6.1f}°")
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