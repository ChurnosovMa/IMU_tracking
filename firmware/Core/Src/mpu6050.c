#include <math.h>
#include "mpu6050.h"

#define DEG_TO_RAD 0.01745329251994329576923690768489

#define WHO_AM_I_REG 0x75
#define PWR_MGMT_1_REG 0x6B
#define SMPLRT_DIV_REG 0x19
#define ACCEL_CONFIG_REG 0x1C
#define ACCEL_XOUT_H_REG 0x3B
#define TEMP_OUT_H_REG 0x41
#define GYRO_CONFIG_REG 0x1B
#define GYRO_XOUT_H_REG 0x43

const uint16_t i2c_timeout = 100;
const double Accel_Z_corrector = 14418.0;


void MPU6050_Calibrate(I2C_HandleTypeDef *I2Cx,
                       MPU6050_t *DataStruct,
                       uint16_t samples)
{
    if (samples == 0)
        return;

    int32_t gx_sum = 0;
    int32_t gy_sum = 0;
    int32_t gz_sum = 0;

    uint8_t Rec_Data[6];

    for (uint16_t i = 0; i < samples; i++)
    {
        if (HAL_I2C_Mem_Read(I2Cx,
                             DataStruct->Address,
                             GYRO_XOUT_H_REG,
                             I2C_MEMADD_SIZE_8BIT,
                             Rec_Data,
                             6,
                             i2c_timeout) != HAL_OK)
        {
            DataStruct->IsCalibrated = 0;
            return;
        }

        gx_sum += (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
        gy_sum += (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
        gz_sum += (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

        HAL_Delay(3);
    }

    DataStruct->Gyro_X_offset = ((float)gx_sum / samples) / 131.0f;
    DataStruct->Gyro_Y_offset = ((float)gy_sum / samples) / 131.0f;
    DataStruct->Gyro_Z_offset = ((float)gz_sum / samples) / 131.0f;

    DataStruct->IsCalibrated = 1;
    DataStruct->dwt_timer = DWT->CYCCNT;
}

uint8_t MPU6050_Init(I2C_HandleTypeDef *I2Cx, MPU6050_t *mpu)
{
    uint8_t check = 0;
    uint8_t Data;
    HAL_StatusTypeDef st;
    // Добавлено дополнительное сбрасывание датчика при нажатии кнопки reset
    Data = 0x80;
    HAL_I2C_Mem_Write(I2Cx, mpu->Address, PWR_MGMT_1_REG, 1, &Data, 1, i2c_timeout);
    HAL_Delay(100); // датчику нужно время подняться после reset
    Data = 0x00;
    HAL_I2C_Mem_Write(I2Cx, mpu->Address, PWR_MGMT_1_REG, 1, &Data, 1, i2c_timeout);

    st = HAL_I2C_Mem_Read(I2Cx, mpu->Address, WHO_AM_I_REG, 1, &check, 1, i2c_timeout);

    if (st == HAL_OK && check == 104)
    {
        Data = 0;
        HAL_I2C_Mem_Write(I2Cx, mpu->Address, PWR_MGMT_1_REG, 1, &Data, 1, i2c_timeout);

        Data = 0x07;
        HAL_I2C_Mem_Write(I2Cx, mpu->Address, SMPLRT_DIV_REG, 1, &Data, 1, i2c_timeout);

        Data = 0x00;
        HAL_I2C_Mem_Write(I2Cx, mpu->Address, ACCEL_CONFIG_REG, 1, &Data, 1, i2c_timeout);

        Data = 0x00;
        HAL_I2C_Mem_Write(I2Cx, mpu->Address, GYRO_CONFIG_REG, 1, &Data, 1, i2c_timeout);

        // --- инициализация кватерниона в "единичное" состояние (без поворота) ---
        mpu->q0 = 1.0f;
        mpu->q1 = 0.0f;
        mpu->q2 = 0.0f;
        mpu->q3 = 0.0f;
        mpu->beta = 0.05f;   // стартовое значение, подбирается экспериментально

        mpu->dwt_timer = DWT->CYCCNT;

        return 0;
    }
    return 1;
}

HAL_StatusTypeDef MPU6050_Read_All(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct)
{
    uint8_t Rec_Data[14];
    int16_t temp;

    HAL_StatusTypeDef status = HAL_I2C_Mem_Read(
        I2Cx, DataStruct->Address, ACCEL_XOUT_H_REG, 1, Rec_Data, 14, i2c_timeout);

    if (status != HAL_OK)
    {
        // связь не удалась — не трогаем DataStruct мусором, сообщаем вызывающему коду
        return status;
    }

    DataStruct->Accel_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
    DataStruct->Accel_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
    DataStruct->Accel_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

    DataStruct->Ax = DataStruct->Accel_X_RAW / 16384.0f;
    DataStruct->Ay = DataStruct->Accel_Y_RAW / 16384.0f;
    DataStruct->Az = DataStruct->Accel_Z_RAW / 16384.0f;

    temp = (int16_t)(Rec_Data[6] << 8 | Rec_Data[7]);
    DataStruct->Gyro_X_RAW = (int16_t)(Rec_Data[8] << 8 | Rec_Data[9]);
    DataStruct->Gyro_Y_RAW = (int16_t)(Rec_Data[10] << 8 | Rec_Data[11]);
    DataStruct->Gyro_Z_RAW = (int16_t)(Rec_Data[12] << 8 | Rec_Data[13]);

    DataStruct->Temperature = (float)((int16_t)temp / (float)340.0 + (float)36.53);

    // ВАЖНО: теперь сразу переводим в рад/с — так нужно для Madgwick
    DataStruct->Gx = ((DataStruct->Gyro_X_RAW / 131.0) - DataStruct->Gyro_X_offset) * DEG_TO_RAD;
    DataStruct->Gy = ((DataStruct->Gyro_Y_RAW / 131.0) - DataStruct->Gyro_Y_offset) * DEG_TO_RAD;
    DataStruct->Gz = ((DataStruct->Gyro_Z_RAW / 131.0) - DataStruct->Gyro_Z_offset) * DEG_TO_RAD;

    // --- расчёт dt per-instance (было глобальным багом при 2+ датчиках) ---
    uint32_t now = DWT->CYCCNT;

    DataStruct->dt = (float)(now - DataStruct->dwt_timer) /
                     (float)SystemCoreClock;

    DataStruct->dwt_timer = now;

    MadgwickAHRSupdateIMU(DataStruct, DataStruct->dt);

    return HAL_OK;
}

void MPU6050_Read_Accel(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct)
{
    uint8_t Rec_Data[6];
    HAL_I2C_Mem_Read(I2Cx, DataStruct->Address, ACCEL_XOUT_H_REG, 1, Rec_Data, 6, i2c_timeout);

    DataStruct->Accel_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
    DataStruct->Accel_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
    DataStruct->Accel_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

    DataStruct->Ax = DataStruct->Accel_X_RAW / 16384.0;
    DataStruct->Ay = DataStruct->Accel_Y_RAW / 16384.0;
    DataStruct->Az = DataStruct->Accel_Z_RAW / Accel_Z_corrector;
}

void MPU6050_Read_Gyro(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct)
{
    uint8_t Rec_Data[6];
    HAL_I2C_Mem_Read(I2Cx, DataStruct->Address, GYRO_XOUT_H_REG, 1, Rec_Data, 6, i2c_timeout);

    DataStruct->Gyro_X_RAW = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
    DataStruct->Gyro_Y_RAW = (int16_t)(Rec_Data[2] << 8 | Rec_Data[3]);
    DataStruct->Gyro_Z_RAW = (int16_t)(Rec_Data[4] << 8 | Rec_Data[5]);

    DataStruct->Gx = ((DataStruct->Gyro_X_RAW / 131.0) - DataStruct->Gyro_X_offset) * DEG_TO_RAD;
    DataStruct->Gy = ((DataStruct->Gyro_Y_RAW / 131.0) - DataStruct->Gyro_Y_offset) * DEG_TO_RAD;
    DataStruct->Gz = ((DataStruct->Gyro_Z_RAW / 131.0) - DataStruct->Gyro_Z_offset) * DEG_TO_RAD;
}

void MPU6050_Read_Temp(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct)
{
    uint8_t Rec_Data[2];
    int16_t temp;
    HAL_I2C_Mem_Read(I2Cx, DataStruct->Address, TEMP_OUT_H_REG, 1, Rec_Data, 2, i2c_timeout);
    temp = (int16_t)(Rec_Data[0] << 8 | Rec_Data[1]);
    DataStruct->Temperature = (float)((int16_t)temp / (float)340.0 + (float)36.53);
}

/*
 * Madgwick AHRS update, IMU-вариант (без магнитометра).
 * Основано на оригинальной публикации S.O.H. Madgwick, 2010.
 * gx,gy,gz — рад/с; ax,ay,az — любые единицы (нормализуются внутри).
 */
void MadgwickAHRSupdateIMU(MPU6050_t *mpu, float dt)
{
    if (dt <= 0.0f || dt > 1.0f) return; // защита от аномального dt (первый вызов/пропуск кадра)

    float q0 = mpu->q0, q1 = mpu->q1, q2 = mpu->q2, q3 = mpu->q3;
    float gx = (float)mpu->Gx, gy = (float)mpu->Gy, gz = (float)mpu->Gz;
    float ax = (float)mpu->Ax, ay = (float)mpu->Ay, az = (float)mpu->Az;

    float recipNorm;
    float s0, s1, s2, s3;
    float qDot1, qDot2, qDot3, qDot4;
    float _2q0, _2q1, _2q2, _2q3, _4q0, _4q1, _4q2, _8q1, _8q2, q0q0, q1q1, q2q2, q3q3;

    // Скорость изменения кватерниона от гироскопа
    qDot1 = 0.5f * (-q1 * gx - q2 * gy - q3 * gz);
    qDot2 = 0.5f * (q0 * gx + q2 * gz - q3 * gy);
    qDot3 = 0.5f * (q0 * gy - q1 * gz + q3 * gx);
    qDot4 = 0.5f * (q0 * gz + q1 * gy - q2 * gx);

    // Коррекция по акселерометру, только если измерение не вырождено (не свободное падение и т.п.)
    if (!((ax == 0.0f) && (ay == 0.0f) && (az == 0.0f)))
    {
        recipNorm = 1.0f / sqrtf(ax * ax + ay * ay + az * az);
        ax *= recipNorm;
        ay *= recipNorm;
        az *= recipNorm;

        _2q0 = 2.0f * q0;
        _2q1 = 2.0f * q1;
        _2q2 = 2.0f * q2;
        _2q3 = 2.0f * q3;
        _4q0 = 4.0f * q0;
        _4q1 = 4.0f * q1;
        _4q2 = 4.0f * q2;
        _8q1 = 8.0f * q1;
        _8q2 = 8.0f * q2;
        q0q0 = q0 * q0;
        q1q1 = q1 * q1;
        q2q2 = q2 * q2;
        q3q3 = q3 * q3;

        // Градиент целевой функции (аналитический якобиан для случая g_ref = [0,0,1])
        s0 = _4q0 * q2q2 + _2q2 * ax + _4q0 * q1q1 - _2q1 * ay;
        s1 = _4q1 * q3q3 - _2q3 * ax + 4.0f * q0q0 * q1 - _2q0 * ay - _4q1 + _8q1 * q1q1 + _8q1 * q2q2 + _4q1 * az;
        s2 = 4.0f * q0q0 * q2 + _2q0 * ax + _4q2 * q3q3 - _2q3 * ay - _4q2 + _8q2 * q1q1 + _8q2 * q2q2 + _4q2 * az;
        s3 = 4.0f * q1q1 * q3 - _2q1 * ax + 4.0f * q2q2 * q3 - _2q2 * ay;

        recipNorm = 1.0f / sqrtf(s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3);
        s0 *= recipNorm;
        s1 *= recipNorm;
        s2 *= recipNorm;
        s3 *= recipNorm;

        // Смешиваем: интеграл гироскопа минус градиентный шаг, взвешенный beta
        qDot1 -= mpu->beta * s0;
        qDot2 -= mpu->beta * s1;
        qDot3 -= mpu->beta * s2;
        qDot4 -= mpu->beta * s3;
    }

    // Интегрирование
    q0 += qDot1 * dt;
    q1 += qDot2 * dt;
    q2 += qDot3 * dt;
    q3 += qDot4 * dt;

    // Нормализация — обязательна на каждом шаге
    recipNorm = 1.0f / sqrtf(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3);
    mpu->q0 = q0 * recipNorm;
    mpu->q1 = q1 * recipNorm;
    mpu->q2 = q2 * recipNorm;
    mpu->q3 = q3 * recipNorm;
}

void Quaternion_ToEuler(const MPU6050_t *mpu, float *roll, float *pitch, float *yaw)
{
    float q0 = mpu->q0, q1 = mpu->q1, q2 = mpu->q2, q3 = mpu->q3;

    *roll  = atan2f(2.0f * (q0 * q1 + q2 * q3), 1.0f - 2.0f * (q1 * q1 + q2 * q2));
    *pitch = asinf(2.0f * (q0 * q2 - q3 * q1));
    *yaw   = atan2f(2.0f * (q0 * q3 + q1 * q2), 1.0f - 2.0f * (q2 * q2 + q3 * q3));

    // в градусы, если удобнее для вывода
    *roll  *= 57.29578f;
    *pitch *= 57.29578f;
    *yaw   *= 57.29578f;
}
