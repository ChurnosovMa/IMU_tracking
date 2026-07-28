/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "i2c.h"
#include "tim.h"
#include "usart.h"
#include "gpio.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mpu6050.h"
#include "core_cm4.h"
#include "BNO_08x_I2C.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */
MPU6050_t MPU6050;
volatile uint8_t BNO_Ready = 0;

float roll1, pitch1, yaw1;
float roll2, pitch2, yaw2;

BNO_RotationVectorWAcc_t quat = {};
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static inline void DWT_Init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static inline void DWT_DelayUs(uint32_t us)
{
    uint32_t start = DWT->CYCCNT;
    uint32_t cycles = us * (SystemCoreClock / 1000000U);
    while ((DWT->CYCCNT - start) < cycles) { }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_I2C1_Init();
  MX_I2C2_Init();
  MX_USART2_UART_Init();
  MX_TIM2_Init();
  /* USER CODE BEGIN 2 */
  MPU6050_t imu2 = {
      .Address = 0x68 << 1
  };

  MPU6050_t imu1 = {
      .Address = 0x69 << 1
  };

  MPU6050_Init(&hi2c1, &imu1);
  MPU6050_Init(&hi2c1, &imu2);

  DWT_Init();
  HAL_TIM_Base_Start(&htim2);
  printf("Start!\r\n");
  printf("INT pin state: %d\r\n",
      HAL_GPIO_ReadPin(BNO_INT_GPIO_Port, BNO_INT_Pin));
  HAL_Delay(100);
// Закоментированная часть с датчиком BNO
  // Вручную сбрасываем сенсор
//  HAL_GPIO_WritePin(BNO_RST_GPIO_Port, BNO_RST_Pin, GPIO_PIN_RESET); // RST = 0
//  HAL_Delay(100);
//  HAL_GPIO_WritePin(BNO_RST_GPIO_Port, BNO_RST_Pin, GPIO_PIN_SET);   // RST = 1
//  HAL_Delay(1000);
//  printf("Сканирование I2C...\r\n");
//  uint8_t found = 0;
//
//  for (uint8_t addr = 1; addr < 128; addr++) {
//      if (HAL_I2C_IsDeviceReady(&hi2c1, addr << 1, 2, 10) == HAL_OK) {
//          printf("Найдено: 0x%02X\r\n", addr);
//          found++;
//      }
//  }
//  if (!found) printf("Устройств не найдено!\r\n");
//  BNO_Ready = 1;
//  HAL_Delay(500);
//  if (BNO_Init() == HAL_OK) {
//      printf("BNO сенсор найден!\r\n");
//      BNO_setFeature(MAGNETIC_FIELD_CALIBRATED, 13333, 0);
//      HAL_Delay(100);
//      BNO_setFeature(ROTATION_VECTOR,           13333, 0);
//      HAL_Delay(100);
//  } else {
//      printf("ОШИБКА: сенсор не отвечает!\r\n");
//      Error_Handler();
//  }
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  MPU6050_Calibrate(&hi2c1, &imu1, 500);
  MPU6050_Calibrate(&hi2c1, &imu2, 500);
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */


	MPU6050_Read_All(&hi2c1, &imu1);
	MPU6050_Read_All(&hi2c1, &imu2);
    char tx[512];

//    sprintf(tx,
//            "%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\r\n",
//			imu1.q0,
//			imu1.q1,
//			imu1.q2,
//			imu1.q3,
//			imu2.q0,
//			imu2.q1,
//			imu2.q2,
//			imu2.q3);


    sprintf(tx,
            "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\r\n",
            imu1.q0, imu1.q1, imu1.q2, imu1.q3,
            imu2.q0, imu2.q1, imu2.q2, imu2.q3
        );

        HAL_UART_Transmit(&huart2, (uint8_t*)tx, strlen(tx), HAL_MAX_DELAY);

//    char msg[] = "UART OK\r\n";
//
//    HAL_UART_Transmit(&huart2,
//                      (uint8_t *)msg,
//                      sizeof(msg) - 1,
//                      HAL_MAX_DELAY);
//

  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
