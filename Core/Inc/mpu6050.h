#ifndef INC_GY521_H_
#define INC_GY521_H_

#include <stdint.h>
#include "i2c.h"


// MPU6050 structure
typedef struct
{
    int16_t Accel_X_RAW;
    int16_t Accel_Y_RAW;
    int16_t Accel_Z_RAW;
    double Ax;
    double Ay;
    double Az;

    int16_t Gyro_X_RAW;
    int16_t Gyro_Y_RAW;
    int16_t Gyro_Z_RAW;
    double Gx;   // рад/с — ВАЖНО: теперь в радианах, не градусах (см. .c)
    double Gy;
    double Gz;

    float Temperature;

    // --- Кватернион ориентации (Madgwick) ---
    float q0, q1, q2, q3;      // w, x, y, z
    float beta;                 // коэффициент доверия акселерометру (0.03..0.1 обычно)
    uint32_t dwt_timer;             // свой таймер per-instance (было глобальным — багфикс)

    uint16_t Address;
    float Gyro_X_offset, Gyro_Y_offset, Gyro_Z_offset;   // в град/с, до перевода в рад
    uint8_t IsCalibrated;
    float dt;
} MPU6050_t;

uint8_t MPU6050_Init(I2C_HandleTypeDef *I2Cx, MPU6050_t *mpu);

void MPU6050_Calibrate(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct, uint16_t samples);

void MPU6050_Read_Accel(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct);

void MPU6050_Read_Gyro(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct);

void MPU6050_Read_Temp(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct);

HAL_StatusTypeDef MPU6050_Read_All(I2C_HandleTypeDef *I2Cx, MPU6050_t *DataStruct);

// Обновление ориентации методом Madgwick (без магнитометра, 6-осевой вариант)
void MadgwickAHRSupdateIMU(MPU6050_t *mpu, float dt);

// Вспомогательное: получить углы Эйлера из кватерниона (для отладки/вывода)
void Quaternion_ToEuler(const MPU6050_t *mpu, float *roll, float *pitch, float *yaw);

#endif /* INC_GY521_H_ */
