/*
 * arm_pose.c - см. arm_pose.h для описания. Прямой перенос с Python,
 * функция в функцию:
 *   quat_conjugate, quat_multiply, quat_normalize, rotate_vector_by_quat,
 *   axis_angle_quat, euler_to_quat, apply_axis_remap, matrix3_to_quat,
 *   v_norm/v_sub/v_scale/v_dot/v_cross, gravity_in_sensor_frame,
 *   axis_from_total_rotation_world, build_world_remap, nutation_angle_deg,
 *   horizontal_project_normalize, compute_precession_deg, compute_body_tilt,
 *   apply_body_correction, compute_shoulder_relative_to_body_deg,
 *   classify_arm_pose - см. arm_viewer_precession.py, эта версия математики
 *   уже проверена численно там, здесь только перевод в C.
 */

#include "arm_pose.h"
#include <math.h>
#include <string.h>

#ifndef ARM_POSE_PI
#define ARM_POSE_PI 3.14159265358979323846f
#endif

/* ------------------------- Базовые типы ------------------------- */

typedef struct { float x, y, z; } Vec3;
typedef ArmPose_Quat Quat;

static const Quat IDENTITY_QUAT = {1.0f, 0.0f, 0.0f, 0.0f};
static const Vec3 DOWN_WORLD = {0.0f, -1.0f, 0.0f};
static const Vec3 UP_WORLD   = {0.0f,  1.0f, 0.0f};
static const Vec3 CHEST_WORLD_DOWN = {0.0f, 0.0f, 1.0f};

/* BASIS_FIX_QUAT / BASIS_FIX_QUAT2 - те же константы, что и в Python,
   найдены и подтверждены экспериментально там - см. историю калибровки. */
static const Quat BASIS_FIX_QUAT  = {0.0f, 0.70710678f, 0.0f, 0.70710678f};
static const Quat BASIS_FIX_QUAT2 = {0.0f, 0.0f, 1.0f, 0.0f};

/* --------------------- Кватернионная/векторная математика --------------------- */

static Quat quat_conj(Quat q) {
    Quat r = { q.w, -q.x, -q.y, -q.z };
    return r;
}

static Quat quat_mul(Quat a, Quat b) {
    Quat r;
    r.w = a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z;
    r.x = a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y;
    r.y = a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x;
    r.z = a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w;
    return r;
}

static Quat quat_normalize(Quat q) {
    float n = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (n < 1e-9f) return IDENTITY_QUAT;
    Quat r = { q.w/n, q.x/n, q.y/n, q.z/n };
    return r;
}

static Vec3 rotate_vec(Quat q, Vec3 v) {
    Quat qv = { 0.0f, v.x, v.y, v.z };
    Quat r = quat_mul(quat_mul(q, qv), quat_conj(q));
    Vec3 out = { r.x, r.y, r.z };
    return out;
}

static Quat axis_angle_quat(Vec3 axis, float angle_deg) {
    float n = sqrtf(axis.x*axis.x + axis.y*axis.y + axis.z*axis.z);
    if (n > 1e-9f) { axis.x /= n; axis.y /= n; axis.z /= n; }
    float a = angle_deg * ARM_POSE_PI / 180.0f;
    float s = sinf(a * 0.5f);
    Quat r = { cosf(a * 0.5f), axis.x*s, axis.y*s, axis.z*s };
    return r;
}

static Quat euler_to_quat(float rx_deg, float ry_deg, float rz_deg) {
    Vec3 ax = {1,0,0}, ay = {0,1,0}, az = {0,0,1};
    Quat qx = axis_angle_quat(ax, rx_deg);
    Quat qy = axis_angle_quat(ay, ry_deg);
    Quat qz = axis_angle_quat(az, rz_deg);
    return quat_mul(qz, quat_mul(qy, qx));
}

static Vec3 v_norm(Vec3 a) {
    float n = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    if (n < 1e-8f) { Vec3 z = {0,0,0}; return z; }
    Vec3 r = { a.x/n, a.y/n, a.z/n };
    return r;
}
static Vec3 v_sub(Vec3 a, Vec3 b) { Vec3 r = {a.x-b.x, a.y-b.y, a.z-b.z}; return r; }
static Vec3 v_scale(Vec3 a, float s) { Vec3 r = {a.x*s, a.y*s, a.z*s}; return r; }
static float v_dot(Vec3 a, Vec3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
static Vec3 v_cross(Vec3 a, Vec3 b) {
    Vec3 r = { a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x };
    return r;
}

static Quat matrix3_to_quat(float m00,float m01,float m02,
                             float m10,float m11,float m12,
                             float m20,float m21,float m22) {
    float trace = m00 + m11 + m22;
    float w,x,y,z,s;
    if (trace > 0.0f) {
        s = 0.5f / sqrtf(trace + 1.0f);
        w = 0.25f / s;
        x = (m21 - m12) * s;
        y = (m02 - m20) * s;
        z = (m10 - m01) * s;
    } else if (m00 > m11 && m00 > m22) {
        s = 2.0f * sqrtf(1.0f + m00 - m11 - m22);
        w = (m21 - m12) / s;
        x = 0.25f * s;
        y = (m01 + m10) / s;
        z = (m02 + m20) / s;
    } else if (m11 > m22) {
        s = 2.0f * sqrtf(1.0f + m11 - m00 - m22);
        w = (m02 - m20) / s;
        x = (m01 + m10) / s;
        y = 0.25f * s;
        z = (m12 + m21) / s;
    } else {
        s = 2.0f * sqrtf(1.0f + m22 - m00 - m11);
        w = (m10 - m01) / s;
        x = (m02 + m20) / s;
        y = (m12 + m21) / s;
        z = 0.25f * s;
    }
    Quat q = { w, x, y, z };
    return quat_normalize(q);
}

/* ------------------------- AXIS_REMAP (руки) ------------------------- */

static Quat g_axis_remap;  /* = euler_to_quat(-90, 90, 0), считается один раз в ArmPose_Init */

static Quat apply_axis_remap(Quat q_raw) {
    return quat_mul(quat_mul(g_axis_remap, q_raw), quat_conj(g_axis_remap));
}

/* --------------------- gravity_in_sensor_frame / нутация --------------------- */

static Vec3 gravity_in_sensor_frame(Quat q_sensor_to_world, Vec3 world_down) {
    Quat q_world_to_sensor = quat_conj(q_sensor_to_world);
    return rotate_vec(q_world_to_sensor, world_down);
}

static float nutation_angle_deg(Quat q, Vec3 local_bone_axis) {
    Vec3 world_dir = rotate_vec(q, local_bone_axis);
    float c = v_dot(world_dir, DOWN_WORLD);
    if (c > 1.0f) c = 1.0f;
    if (c < -1.0f) c = -1.0f;
    return acosf(c) * 180.0f / ARM_POSE_PI;
}

static Vec3 horizontal_project_normalize(Vec3 v) {
    float proj = v_dot(v, UP_WORLD);
    Vec3 horiz = v_sub(v, v_scale(UP_WORLD, proj));
    return v_norm(horiz);
}

/* Знаковый угол поворота q вокруг axis - см. twist_angle_about_axis_deg в
   Python. Для датчика груди axis ДОЛЖНА быть CHEST_WORLD_DOWN. */
static float twist_angle_about_axis_deg(Quat q, Vec3 axis) {
    float proj = q.x*axis.x + q.y*axis.y + q.z*axis.z;
    return 2.0f * atan2f(proj, q.w) * 180.0f / ARM_POSE_PI;
}

/* --------------------- ось из суммарного поворота (калибровка L) --------------------- */

/* Возвращает 1 при успехе (norm достаточна), 0 если движение слишком мало
   (аналог None в Python). */
static int axis_from_total_rotation_world(Quat q1, Quat q2, Vec3 *out_axis) {
    Quat dq = quat_mul(q2, quat_conj(q1));
    float w = dq.w, x = dq.x, y = dq.y, z = dq.z;
    if (w < 0.0f) { x = -x; y = -y; z = -z; }
    float n = sqrtf(x*x + y*y + z*z);
    if (n < 1e-4f) return 0;
    out_axis->x = x/n; out_axis->y = y/n; out_axis->z = z/n;
    return 1;
}

/* --------------------- build_world_remap --------------------- */

static Quat build_world_remap(Vec3 down_axis, Vec3 pitch_axis) {
    Vec3 d = v_norm(down_axis);
    float proj = v_dot(d, pitch_axis);
    Vec3 f = v_norm(v_sub(pitch_axis, v_scale(d, proj)));
    Vec3 l = v_cross(d, f);

    Vec3 L_scene = {-1.0f, 0.0f, 0.0f};
    Vec3 D_scene = DOWN_WORLD;
    Vec3 F_scene = v_cross(L_scene, D_scene);

    Quat q_measured = matrix3_to_quat(l.x, d.x, f.x,
                                       l.y, d.y, f.y,
                                       l.z, d.z, f.z);
    Quat q_scene = matrix3_to_quat(L_scene.x, D_scene.x, F_scene.x,
                                    L_scene.y, D_scene.y, F_scene.y,
                                    L_scene.z, D_scene.z, F_scene.z);
    return quat_mul(q_scene, quat_conj(q_measured));
}

/* --------------------- apply_body_correction --------------------- */

static Quat apply_body_correction(Quat q_raw, Quat R_segment, Quat q_body_scene_raw) {
    Quat q_seg_scene = quat_mul(quat_mul(R_segment, q_raw), quat_conj(R_segment));
    Quat q_seg_scene_corr = quat_mul(quat_conj(q_body_scene_raw), q_seg_scene);
    return quat_mul(quat_mul(quat_conj(R_segment), q_seg_scene_corr), R_segment);
}

/* ============================================================
 *                    СОСТОЯНИЕ КАЛИБРОВКИ
 * ============================================================ */

typedef enum { CAP_NONE = 0, CAP_G, CAP_L } CaptureState;
static CaptureState s_capture = CAP_NONE;
static uint32_t s_capture_start_ms = 0;

/* --- калибровка G "рука вниз" --- */
static Vec3 s_local_bone_axis_upper;
static Vec3 s_local_bone_axis_fore;
static int  s_is_calibrated_down = 0;
static Vec3 s_g_sum_upper, s_g_sum_fore;   /* накопление суммы за время записи */
static uint32_t s_g_sample_count = 0;

/* --- калибровка F "рука вперёд" --- */
static Vec3 s_y_group;             /* "вперёд" для группы, в СК плеча */
static Vec3 s_x_group;             /* "влево" для группы, в СК плеча */
static Quat s_q_chest_cal;         /* точка отсчёта поворота тела */
static int  s_is_calibrated_forward = 0;

/* --- калибровка L "наклон" - ИЗМЕНЕНО: теперь R_arm_remap/R_forearm_remap
   строятся ЗДЕСЬ, из ОБЩЕГО с телом движения, а не в F из Y_group - тот
   способ (в предположении 'рука в момент F смотрела туда же, куда тело')
   давал непредсказуемо большую ошибку (до ~34° в тесте 'лечь, рука не
   двигается относительно тела' - недопустимо, см. историю Python). ---*/
static Quat s_q_chest_lean_start;
static Quat s_q_upper_lean_start;   /* ДОБАВЛЕНО */
static Quat s_q_fore_lean_start;    /* ДОБАВЛЕНО */
static int  s_is_world_remap_calibrated = 0;
static Quat s_R_world_remap;       /* С BASIS_FIX - для отображения/наклона */
static Quat s_R_world_remap_raw;   /* БЕЗ BASIS_FIX - для коррекции руки (см. Python) */
static Quat s_R_arm_remap;         /* СК датчика плеча -> сцена, из общего движения L */
static int  s_is_arm_remap_calibrated = 0;
static Quat s_R_forearm_remap;     /* ДОБАВЛЕНО - то же для предплечья */
static int  s_is_forearm_remap_calibrated = 0;  /* ДОБАВЛЕНО */

/* --- последние сырые показания (после AXIS_REMAP для руки) --- */
static Quat s_q_upper, s_q_fore, s_q_chest;
static int  s_have_data = 0;

/* --- посчитанные углы (кэш последнего кадра) --- */
static ArmPose_Angles s_angles;
static ArmPose_RelativeAngles s_rel_angles;  /* ДОБАВЛЕНО */
static uint8_t s_pose_number = 6;

/* ============================================================
 *                    КАЛИБРОВКА - РЕАЛИЗАЦИЯ
 * ============================================================ */

void ArmPose_Init(void) {
    g_axis_remap = euler_to_quat(-90.0f, 90.0f, 0.0f);
    s_is_calibrated_down = 0;
    s_is_calibrated_forward = 0;
    s_is_arm_remap_calibrated = 0;
    s_is_world_remap_calibrated = 0;
    s_capture = CAP_NONE;
    s_have_data = 0;
    s_angles.shoulder_deg = NAN;
    s_angles.elbow_deg = NAN;
    s_angles.precession_deg = NAN;
    s_angles.tilt_fwd_deg = NAN;
    s_angles.tilt_side_deg = NAN;
    s_rel_angles.shoulder_deg = NAN;
    s_rel_angles.elbow_deg = NAN;
    s_rel_angles.azimuth_deg = NAN;
    s_pose_number = 6;
}

static void start_stop_capture_down(uint32_t tick_ms) {
    if (s_capture == CAP_NONE) {
        s_capture = CAP_G;
        s_capture_start_ms = tick_ms;
        s_g_sum_upper.x = s_g_sum_upper.y = s_g_sum_upper.z = 0.0f;
        s_g_sum_fore.x  = s_g_sum_fore.y  = s_g_sum_fore.z  = 0.0f;
        s_g_sample_count = 0;
        return;
    }
    if (s_capture != CAP_G) return; /* заняты другой записью */
    s_capture = CAP_NONE;
    if (s_g_sample_count < 5) return; /* слишком мало сэмплов, как в Python */

    Vec3 mean_upper = v_scale(s_g_sum_upper, 1.0f / (float)s_g_sample_count);
    Vec3 mean_fore  = v_scale(s_g_sum_fore,  1.0f / (float)s_g_sample_count);
    s_local_bone_axis_upper = v_norm(mean_upper);
    s_local_bone_axis_fore  = v_norm(mean_fore);
    s_is_calibrated_down = 1;
}

static void try_calibrate_forward(void) {
    /* ИЗМЕНЕНО: раньше здесь ЕЩЁ строился s_R_arm_remap (из Y_group) - это
       давало ненадёжную ошибку, см. пояснение у объявления s_R_arm_remap
       выше. Теперь F отвечает только за s_q_chest_cal (точка отсчёта
       поворота тела) и s_y_group/s_x_group (опорное направление азимута) -
       s_R_arm_remap/s_R_forearm_remap строятся в start_stop_capture_lean(). */
    if (!s_is_calibrated_down || !s_have_data) return;

    Vec3 bone_dir_own = rotate_vec(s_q_upper, s_local_bone_axis_upper);
    Vec3 y_group = horizontal_project_normalize(bone_dir_own);
    if (y_group.x == 0.0f && y_group.y == 0.0f && y_group.z == 0.0f) {
        return; /* рука вертикально - нельзя определить азимут, как в Python */
    }
    Vec3 x_group = v_norm(v_cross(UP_WORLD, y_group));

    s_y_group = y_group;
    s_x_group = x_group;
    s_q_chest_cal = s_q_chest;
    s_is_calibrated_forward = 1;
}

static void reset_calibration_forward(void) {
    s_is_calibrated_forward = 0;
}

static void start_stop_capture_lean(uint32_t tick_ms) {
    if (s_capture == CAP_NONE) {
        s_capture = CAP_L;
        s_capture_start_ms = tick_ms;
        s_q_chest_lean_start = s_q_chest;
        s_q_upper_lean_start = s_q_upper;  /* ДОБАВЛЕНО */
        s_q_fore_lean_start  = s_q_fore;   /* ДОБАВЛЕНО */
        return;
    }
    if (s_capture != CAP_L) return;
    s_capture = CAP_NONE;

    Vec3 axis;
    if (!axis_from_total_rotation_world(s_q_chest_lean_start, s_q_chest, &axis)) {
        return; /* слишком мало движения, как в Python */
    }
    Quat basis_fix = quat_mul(BASIS_FIX_QUAT2, BASIS_FIX_QUAT);
    s_R_world_remap_raw = build_world_remap(CHEST_WORLD_DOWN, axis);
    s_R_world_remap = quat_mul(basis_fix, s_R_world_remap_raw);
    s_is_world_remap_calibrated = 1;

    /* ДОБАВЛЕНО: то же самое движение (то же начало/конец записи) - для
       руки и предплечья, см. пояснение у объявления s_R_arm_remap выше. */
    Vec3 upper_axis;
    if (axis_from_total_rotation_world(s_q_upper_lean_start, s_q_upper, &upper_axis)) {
        s_R_arm_remap = build_world_remap(DOWN_WORLD, upper_axis);
        s_is_arm_remap_calibrated = 1;
    } else {
        s_is_arm_remap_calibrated = 0;
    }

    Vec3 fore_axis;
    if (axis_from_total_rotation_world(s_q_fore_lean_start, s_q_fore, &fore_axis)) {
        s_R_forearm_remap = build_world_remap(DOWN_WORLD, fore_axis);
        s_is_forearm_remap_calibrated = 1;
    } else {
        s_is_forearm_remap_calibrated = 0;
    }
}

void ArmPose_HandleCommand(char cmd) {
    /* tick_ms передаётся через ArmPose_Update по ходу работы; здесь для
       start_stop_capture_* берём "последний известный" момент - см.
       ArmPose_Update, где хранится s_last_tick_ms. */
    extern uint32_t ArmPose_LastTickMs(void);
    uint32_t now = ArmPose_LastTickMs();
    switch (cmd) {
        case 'g': start_stop_capture_down(now); break;
        case 'f': try_calibrate_forward(); break;
        case 'l': start_stop_capture_lean(now); break;
        case 'r': reset_calibration_forward(); break;
        default: break;
    }
}

/* ============================================================
 *                    РАСЧЁТ УГЛОВ (каждый кадр)
 * ============================================================ */

static float compute_precession_deg(void) {
    if (!s_is_calibrated_forward) return NAN;

    Vec3 bone_dir_own = rotate_vec(s_q_upper, s_local_bone_axis_upper);
    Vec3 horiz_now = horizontal_project_normalize(bone_dir_own);
    float cos_a = v_dot(horiz_now, s_y_group);
    float sin_a = v_dot(horiz_now, s_x_group);
    float azimuth_arm_deg = atan2f(sin_a, cos_a) * 180.0f / ARM_POSE_PI;

    Quat q_body_delta = quat_mul(s_q_chest, quat_conj(s_q_chest_cal));
    float body_yaw_since_cal_deg = twist_angle_about_axis_deg(q_body_delta, CHEST_WORLD_DOWN);

    /* ЗНАК МЕНЯЛСЯ ДВАЖДЫ - см. подробный докстринг compute_precession_deg в
       Python. Синтетическая проверка (произвольные условные крепления
       датчиков) указывала на сложение, но на РЕАЛЬНОМ железе пользователь
       подтвердил тестом, что верно именно ВЫЧИТАНИЕ (как было изначально) -
       синтетическая проверка не воспроизводила реальную связку AXIS_REMAP +
       фактическое крепление датчиков. */
    float total_deg = azimuth_arm_deg - body_yaw_since_cal_deg;
    /* приводим к (-180, 180] */
    total_deg = fmodf(total_deg + 180.0f, 360.0f);
    if (total_deg < 0.0f) total_deg += 360.0f;
    total_deg -= 180.0f;
    return total_deg;
}

/* pitch/roll наклона тела - применяем q_body_scene к СОБСТВЕННОЙ вертикали
   сцены (0,1,0), см. фикс бага в Python (не путать CHEST_WORLD_DOWN с
   локальным вектором тела!). */
static void compute_body_tilt(float *out_pitch, float *out_roll) {
    if (!s_is_world_remap_calibrated || !s_is_calibrated_forward) {
        *out_pitch = NAN; *out_roll = NAN; return;
    }
    Quat body_delta = quat_mul(s_q_chest, quat_conj(s_q_chest_cal));
    Quat q_body_scene = quat_mul(quat_mul(s_R_world_remap, body_delta), quat_conj(s_R_world_remap));
    Vec3 up_axis = {0.0f, 1.0f, 0.0f};
    Vec3 up_now = rotate_vec(q_body_scene, up_axis);
    float vertical = up_now.y;
    float forward  = -up_now.z;
    float sideways =  up_now.x;
    *out_pitch = atan2f(forward, vertical) * 180.0f / ARM_POSE_PI;
    *out_roll  = atan2f(sideways, vertical) * 180.0f / ARM_POSE_PI;
}

/* Заменяет compute_shoulder_relative_to_body_deg() - см. Python
   compute_arm_angles_relative_to_body(). Пишет результат через указатели
   (любой может остаться NAN, если соответствующая часть калибровки не
   готова - локоть отдельно требует s_is_forearm_remap_calibrated). */
static void compute_arm_angles_relative_to_body(float *out_shoulder, float *out_elbow, float *out_azimuth) {
    *out_shoulder = NAN; *out_elbow = NAN; *out_azimuth = NAN;
    if (!s_is_calibrated_down || !s_is_arm_remap_calibrated || !s_is_world_remap_calibrated
        || !s_is_calibrated_forward) {
        return;
    }
    Quat body_delta_raw = quat_mul(s_q_chest, quat_conj(s_q_chest_cal));
    Quat q_body_scene_raw = quat_mul(quat_mul(s_R_world_remap_raw, body_delta_raw),
                                      quat_conj(s_R_world_remap_raw));
    Quat q_upper_corrected = apply_body_correction(s_q_upper, s_R_arm_remap, q_body_scene_raw);
    float shoulder_deg = nutation_angle_deg(q_upper_corrected, s_local_bone_axis_upper);
    *out_shoulder = shoulder_deg;

    if (s_is_forearm_remap_calibrated) {
        Quat q_fore_corrected = apply_body_correction(s_q_fore, s_R_forearm_remap, q_body_scene_raw);
        float forearm_abs_deg = nutation_angle_deg(q_fore_corrected, s_local_bone_axis_fore);
        *out_elbow = forearm_abs_deg - shoulder_deg;
    }

    Vec3 bone_dir_corrected = rotate_vec(q_upper_corrected, s_local_bone_axis_upper);
    Vec3 horiz_corrected = horizontal_project_normalize(bone_dir_corrected);
    if (!(horiz_corrected.x == 0.0f && horiz_corrected.y == 0.0f && horiz_corrected.z == 0.0f)) {
        float cos_a = v_dot(horiz_corrected, s_y_group);
        float sin_a = v_dot(horiz_corrected, s_x_group);
        *out_azimuth = atan2f(sin_a, cos_a) * 180.0f / ARM_POSE_PI;
    }
}

/* --------------------- classify_arm_pose --------------------- */
/* Прямой порт Python classify_arm_pose() (версия ПОЛЬЗОВАТЕЛЯ - добавлена
   категория 'рука согнута', 'прочее' сдвинуто на 5, диапазон прецессии для
   'рука вбок' ОТРИЦАТЕЛЬНЫЙ [-100,-60], не [60,100] - подтверждено на
   практике, не переспрашиваю). Возвращает 1-5 (реальные категории) или
   6, если ещё не готово (нет G+F минимум) - "не откалибровано" пришлось
   сдвинуть с 5 на 6, т.к. 5 теперь занята категорией "прочее". tilt может
   быть NAN (L ещё не сделана) - тогда tilt_ok=1 по умолчанию (как в Python).
   ВАЖНО: elbow_deg = forearm_abs_deg - shoulder_deg - для ПРЯМОЙ
   (несогнутой) руки это значение БЛИЗКО К НУЛЮ, а не к 180° - проверено
   численно на самой калибровочной позе G ("рука вниз, локоть прямой"). */
static uint8_t classify_arm_pose(float shoulder_deg, float elbow_deg,
                                  float precession_deg, float tilt_fwd, float tilt_side) {
    if (isnan(shoulder_deg) || isnan(elbow_deg)) return 6;

    int tilt_ok = 1;
    if (!isnan(tilt_fwd) && !isnan(tilt_side)) {
        tilt_ok = (fabsf(tilt_fwd) < 25.0f) && (fabsf(tilt_side) < 25.0f);
    }

    if ((180.0f - elbow_deg) < 110.0f && tilt_ok) {
        return 4; /* рука согнута */
    }

    if (shoulder_deg < 15.0f && (180.0f - elbow_deg) > 160.0f && tilt_ok) {
        return 1; /* рука вниз */
    }

    if (shoulder_deg > 60.0f && (180.0f - elbow_deg) > 160.0f && tilt_ok && !isnan(precession_deg)) {
        if (precession_deg >= -30.0f && precession_deg <= 30.0f) return 2; /* рука вперёд */
        if (precession_deg >= -100.0f && precession_deg <= -60.0f) return 3; /* рука вбок */
    }

    return 5; /* прочее положение */
}

/* ============================================================
 *                    ГЛАВНЫЙ ВХОД - ArmPose_Update
 * ============================================================ */

static uint32_t s_last_tick_ms = 0;
uint32_t ArmPose_LastTickMs(void) { return s_last_tick_ms; }

void ArmPose_Update(ArmPose_Quat q_upper_raw, ArmPose_Quat q_fore_raw,
                     ArmPose_Quat q_chest_raw, uint32_t tick_ms) {
    s_last_tick_ms = tick_ms;

    s_q_upper = apply_axis_remap(q_upper_raw);
    s_q_fore  = apply_axis_remap(q_fore_raw);
    s_q_chest = q_chest_raw; /* грудь НЕ пропускаем через AXIS_REMAP руки, см. Python */
    s_have_data = 1;

    /* накопление данных калибровки G, если идёт запись */
    if (s_capture == CAP_G) {
        Vec3 g_up = gravity_in_sensor_frame(s_q_upper, DOWN_WORLD);
        Vec3 g_fo = gravity_in_sensor_frame(s_q_fore, DOWN_WORLD);
        s_g_sum_upper.x += g_up.x; s_g_sum_upper.y += g_up.y; s_g_sum_upper.z += g_up.z;
        s_g_sum_fore.x  += g_fo.x; s_g_sum_fore.y  += g_fo.y; s_g_sum_fore.z  += g_fo.z;
        s_g_sample_count++;
        /* подстраховка - автозавершение, если забыли прислать второе 'g' */
        if (tick_ms - s_capture_start_ms > ARM_POSE_G_CAPTURE_MAX_MS) {
            start_stop_capture_down(tick_ms); /* второй вызов = "стоп" */
        }
    }
    /* для CAP_L отдельного накопления не нужно - нужен только последний
       s_q_chest на момент второго 'l', он уже обновлён выше */

    /* --- углы --- */
    if (s_is_calibrated_down) {
        s_angles.shoulder_deg = nutation_angle_deg(s_q_upper, s_local_bone_axis_upper);
        float forearm_abs = nutation_angle_deg(s_q_fore, s_local_bone_axis_fore);
        s_angles.elbow_deg = forearm_abs - s_angles.shoulder_deg;
    } else {
        s_angles.shoulder_deg = NAN;
        s_angles.elbow_deg = NAN;
    }

    s_angles.precession_deg = compute_precession_deg();
    compute_body_tilt(&s_angles.tilt_fwd_deg, &s_angles.tilt_side_deg);

    /* ИЗМЕНЕНО: используем скорректированные (относительно тела) значения
       для классификации, где они уже доступны (нужны F+L) - иначе
       откатываемся на абсолютные (относительно истинной мировой
       вертикали), как раньше, чтобы классификация работала сразу после G -
       см. Python main(), тот же приём (shoulder_for_pose и т.п.). */
    float shoulder_rel, elbow_rel, azimuth_rel;
    compute_arm_angles_relative_to_body(&shoulder_rel, &elbow_rel, &azimuth_rel);
    s_rel_angles.shoulder_deg = shoulder_rel;
    s_rel_angles.elbow_deg = elbow_rel;
    s_rel_angles.azimuth_deg = azimuth_rel;
    float shoulder_for_pose = isnan(shoulder_rel) ? s_angles.shoulder_deg : shoulder_rel;
    float elbow_for_pose = isnan(elbow_rel) ? s_angles.elbow_deg : elbow_rel;
    float azimuth_for_pose = isnan(azimuth_rel) ? s_angles.precession_deg : azimuth_rel;

    s_pose_number = classify_arm_pose(shoulder_for_pose, elbow_for_pose,
                                       azimuth_for_pose,
                                       s_angles.tilt_fwd_deg, s_angles.tilt_side_deg);
}

uint8_t ArmPose_GetPoseNumber(void) { return s_pose_number; }
ArmPose_Angles ArmPose_GetAngles(void) { return s_angles; }
ArmPose_RelativeAngles ArmPose_GetRelativeAngles(void) { return s_rel_angles; }
