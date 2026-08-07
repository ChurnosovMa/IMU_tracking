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
#include "arm_pose.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define BNO_REPORT_INTERVAL_US  10000U   // 10 ms -> 100 Hz rotation-vector reports
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

/* --- калибровка/классификация положения руки (arm_pose.c), команды 'g'/'f'/'l'/'r'
   принимаются побайтово по UART2 (тот же порт, что используется для передачи
   данных на ПК) --- */
static volatile uint8_t s_rx_cmd_byte = 0;
static volatile uint8_t s_rx_cmd_pending = 0;
static uint32_t s_tick_ms = 0;  /* грубый счётчик мс, инкрементируется в основном цикле - см. ниже */
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

// BNO_INT (PA9) drives this HAL callback; it is what wakes waitInt() inside the BNO driver.
// If your project already defines HAL_GPIO_EXTI_Callback elsewhere, merge this into it
// instead of defining it twice (the linker will reject duplicates).
void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
    if (GPIO_Pin == BNO_INT_Pin)
    {
        BNO_Ready = 1;
    }
}

/* Однобайтовый асинхронный приём команд калибровки с ПК ('g','f','l','r').
   ВАЖНО: если у вас уже ЕСТЬ обработчик HAL_UART_RxCpltCallback в проекте -
   слейте это тело в него (линковщик не даст определить функцию дважды),
   так же как и с HAL_GPIO_EXTI_Callback выше. */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART2)
    {
        s_rx_cmd_pending = 1;
        /* перезапускаем приём следующего байта */
        HAL_UART_Receive_IT(&huart2, (uint8_t*)&s_rx_cmd_byte, 1);
    }
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

  uint8_t bnoReady = 0;
  if (BNO_Init() == HAL_OK)
  {
      printf("BNO08x found!\r\n");
      // Absolute fused orientation, already sensor-calibrated on-chip
      if (BNO_setFeature(ROTATION_VECTOR, BNO_REPORT_INTERVAL_US, 0) == HAL_OK)
      {
          bnoReady = 1;
      }
      else
      {
          printf("BNO: failed to enable ROTATION_VECTOR\r\n");
      }
  }
  else
  {
      printf("BNO: sensor not responding!\r\n");
  }

  ArmPose_Init();
  HAL_UART_Receive_IT(&huart2, (uint8_t*)&s_rx_cmd_byte, 1);  /* запускаем приём команд калибровки */
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

	if (bnoReady && BNO_dataAvailable() == HAL_OK)
	{
	    if (BNO_getSensorEventID() == ROTATION_VECTOR)
	    {
	        quat = getRotationVector();
	    }
	}

	/* обработка команды калибровки, если пришла с ПК ('g'/'f'/'l'/'r') */
	if (s_rx_cmd_pending)
	{
	    s_rx_cmd_pending = 0;
	    ArmPose_HandleCommand((char)s_rx_cmd_byte);
	}

	/* грубая оценка времени в мс - HAL_GetTick() уже даёт готовый счётчик
	   (инкрементируется системным таймером HAL, 1мс/тик по умолчанию) -
	   используем его напрямую, s_tick_ms сохранён только для наглядности */
	s_tick_ms = HAL_GetTick();

	ArmPose_Quat q_upper = { imu1.q0, imu1.q1, imu1.q2, imu1.q3 };
	ArmPose_Quat q_fore  = { imu2.q0, imu2.q1, imu2.q2, imu2.q3 };
	ArmPose_Quat q_chest = { quat.Real, quat.I, quat.J, quat.K };
	ArmPose_Update(q_upper, q_fore, q_chest, s_tick_ms);
	uint8_t pose_number = ArmPose_GetPoseNumber();

    char tx[512];
    sprintf(tx,
            "%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %d\r\n",
            imu1.q0, imu1.q1, imu1.q2, imu1.q3,
            imu2.q0, imu2.q1, imu2.q2, imu2.q3,
			quat.Real, quat.I, quat.J, quat.K,
			(int)pose_number
        );

    HAL_UART_Transmit(&huart2, (uint8_t*)tx, strlen(tx), HAL_MAX_DELAY);
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
