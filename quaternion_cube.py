"""
Приём кватерниона с BNO-сенсора по UART и визуализация вращающегося куба.

Установка зависимостей:
    pip install pyserial pygame PyOpenGL PyOpenGL_accelerate

Формат данных, который ожидается от сенсора (по умолчанию):
    Текстовая строка вида "w,x,y,z\n", например:
        1.0000,0.0000,0.0000,0.0000\n

Если твой сенсор шлёт данные в другом порядке (например x,y,z,w) —
поменяй порядок присваивания в функции parse_quaternion().
Если данные бинарные — см. комментарий в конце файла (BINARY MODE).
"""

import sys
import threading
import time

import serial
import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT
from OpenGL.GL import *
from OpenGL.GLU import *

# ======================= НАСТРОЙКИ =======================
SERIAL_PORT = "COM3"       # Windows: "COM3", "COM4"...  Linux/Mac: "/dev/ttyUSB0", "/dev/tty.usbserial-XXXX"
BAUDRATE = 115200
# ===========================================================

# Глобальное состояние кватерниона (общее между потоками)
quat_lock = threading.Lock()
current_quat = [1.0, 0.0, 0.0, 0.0]  # w, x, y, z (нейтральный поворот)
running = True


def parse_quaternion(line: str):
    """Парсит строку 'w,x,y,z' и возвращает кортеж float или None при ошибке."""
    try:
        parts = line.strip().split(",")
        if len(parts) != 4:
            return None
        w, x, y, z = (float(p) for p in parts)
        return (w, x, y, z)
    except ValueError:
        return None


def serial_reader_thread():
    """Фоновый поток: читает строки из UART и обновляет current_quat."""
    global current_quat, running

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
            q = parse_quaternion(line)
            if q is not None:
                with quat_lock:
                    current_quat = q
        except Exception as e:
            print(f"Ошибка чтения serial: {e}")
            time.sleep(0.1)

    ser.close()


def quaternion_to_matrix(w, x, y, z):
    """Конвертирует кватернион в матрицу поворота 4x4 (формат для glMultMatrixf, column-major)."""
    n = w * w + x * x + y * y + z * z
    if n < 1e-8:
        return [1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1]
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z

    # OpenGL матрицы column-major -> заполняем в нужном порядке
    m = [
        1 - (yy + zz), xy + wz,       xz - wy,       0,
        xy - wz,       1 - (xx + zz), yz + wx,       0,
        xz + wy,       yz - wx,       1 - (xx + yy), 0,
        0,             0,             0,             1,
    ]
    return m


def draw_cube():
    vertices = (
        (1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1),
        (1, -1, 1), (1, 1, 1), (-1, -1, 1), (-1, 1, 1),
    )
    edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 7), (3, 6),
    )
    faces = (
        (0, 1, 2, 3), (4, 5, 7, 6), (0, 1, 5, 4),
        (2, 3, 6, 7), (1, 2, 7, 5), (0, 3, 6, 4),
    )
    colors = (
        (0.8, 0.2, 0.2), (0.2, 0.8, 0.2), (0.2, 0.2, 0.8),
        (0.8, 0.8, 0.2), (0.8, 0.2, 0.8), (0.2, 0.8, 0.8),
    )

    glBegin(GL_QUADS)
    for face, color in zip(faces, colors):
        glColor3fv(color)
        for vertex_idx in face:
            glVertex3fv(vertices[vertex_idx])
    glEnd()

    glColor3f(0, 0, 0)
    glBegin(GL_LINES)
    for edge in edges:
        for vertex_idx in edge:
            glVertex3fv(vertices[vertex_idx])
    glEnd()


def main():
    global running

    reader = threading.Thread(target=serial_reader_thread, daemon=True)
    reader.start()

    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("BNO Quaternion Cube Viewer")

    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -6)
    glEnable(GL_DEPTH_TEST)

    clock = pygame.time.Clock()

    try:
        while running:
            for event in pygame.event.get():
                if event.type == QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            with quat_lock:
                w, x, y, z = current_quat

            glPushMatrix()
            rot_matrix = quaternion_to_matrix(w, x, y, z)
            glMultMatrixf(rot_matrix)
            draw_cube()
            glPopMatrix()

            pygame.display.flip()
            clock.tick(60)
    finally:
        running = False
        pygame.quit()


if __name__ == "__main__":
    main()


# ============================================================
# BINARY MODE (если сенсор шлёт бинарные данные, а не текст):
#
# Например, если BNO055/BNO085 шлёт 4 float по 4 байта (little-endian)
# без разделителей, замени serial_reader_thread на что-то вроде:
#
#     import struct
#     while running:
#         raw = ser.read(16)  # 4 * float32
#         if len(raw) == 16:
#             w, x, y, z = struct.unpack('<ffff', raw)
#             with quat_lock:
#                 current_quat = (w, x, y, z)
#
# Также часто перед данными идёт байт-заголовок (например 0xAA) —
# тогда сначала нужно синхронизироваться по этому байту.
# ============================================================
