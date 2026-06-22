import numpy as np
from scipy.optimize import minimize
import random

# --- 1. ФИЗИЧЕСКИЕ КОНСТАНТЫ И НАСТРОЙКИ СЕТИ ---
SPEED_OF_SOUND = 343.0  # Скорость звука в воздухе (м/с) при ~20°C

# Координаты трех соседних столбов ЛЭП (X, Y) в метрах.
# Столбы стоят по линии ЛЭП вдоль оси X каждые 100 метров.
STATIONS = np.array([
    [0.0,   0.0],    # Столб №1 (Базовый)
    [100.0, 0.0],    # Столб №2
    [200.0, 0.0]     # Столб №3
])

# --- 2. СИМУЛЯЦИЯ ПОЛЕТА ДРОНА (ГЕНЕРАЦИЯ ДАННЫХ) ---
def simulate_drone_flight(true_drone_pos, wind_noise_level=1e-5):
    """
    Симулирует время прихода звука от дрона на каждый столб ЛЭП.
    Добавляет случайный шум (помехи от ветра/гула проводов).
    """
    arrival_times = []
    for station in STATIONS:
        # Считаем точное расстояние от дрона до столба
        distance = np.linalg.norm(true_drone_pos - station)
        # Время, за которое звук долетит до микрофона
        pure_time = distance / SPEED_OF_SOUND
        # Добавляем микросекундный шум ветра
        noise = random.gauss(0, wind_noise_level)
        arrival_times.append(pure_time + noise)
    
    return np.array(arrival_times)

# --- 3. МАТЕМАТИЧЕСКИЙ АЛГОРИТМ ТРИАНГУЛЯЦИИ (ИИ-ЯДРО) ---
def tdoa_loss_function(estimated_pos, arrival_times):
    """
    Функция ошибки. Сравнивает гипотетические задержки времени 
    для точки `estimated_pos` с реальными задержками, полученными от столбов.
    """
    x, y = estimated_pos
    # Считаем теоретические расстояния от гипотетической точки до каждого столба
    d0 = np.sqrt(x**2 + y**2)
    d1 = np.sqrt((x - STATIONS[1][0])**2 + y**2)
    d2 = np.sqrt((x - STATIONS[2][0])**2 + y**2)
    
    # Теоретическая разница во времени между столбами (относительно Столба №1)
    theoretical_tdoa1 = (d1 - d0) / SPEED_OF_SOUND
    theoretical_tdoa2 = (d2 - d0) / SPEED_OF_SOUND
    
    # Реальная разница во времени, зафиксированная датчиками
    real_tdoa1 = arrival_times[1] - arrival_times[0]
    real_tdoa2 = arrival_times[2] - arrival_times[0]
    
    # Возвращаем сумму квадратов невязок (ошибку)
    error = (theoretical_tdoa1 - real_tdoa1)**2 + (theoretical_tdoa2 - real_tdoa2)**2
    return error

def locate_drone(arrival_times):
    """
    Запускает поиск точки на плоскости, которая минимизирует ошибку TDOA.
    """
    # Стартовая догадка алгоритма (центр нашей сетки столбов)
    initial_guess = np.array([100.0, 50.0])
    
    # Находим глобальный минимум функции ошибки методом Nelder-Mead
    result = minimize(tdoa_loss_function, initial_guess, args=(arrival_times,), method='Nelder-Mead')
    return result.x

# --- 4. ДЕМОНСТРАЦИОННЫЙ ЗАПУСК ---
if __name__ == "__main__":
    print("⚡ СИСТЕМА FIBER SONIC SENTINEL: ТЕСТИРОВАНИЕ АЛГОРИТМА ТРИАНГУЛЯЦИИ ⚡\n")
    print(f"Положение столбов ЛЭП:\n Столб 1: {STATIONS[0]}\n Столб 2: {STATIONS[1]}\n Столб 3: {STATIONS[2]}\n")
    
    # Задаем реальное положение подлетающего дрона:
    # Он летит на расстоянии 45 метров вбок от ЛЭП и на отметке 125 метров вдоль линии
    TRUE_DRONE_POSITION = np.array([125.0, 45.0])
    print(f"[СИМУЛЯЦИЯ] Дрон реально находится в точке: X = {TRUE_DRONE_POSITION[0]}м, Y = {TRUE_DRONE_POSITION[1]}м")
    
    # Шаг 1: Датчики на столбах ловят звук с учетом шума ветра
    sensor_data = simulate_drone_flight(TRUE_DRONE_POSITION, wind_noise_level=1e-6)
    print(f"[ДАТЧИКИ] Время фиксации звука столбами (сек):")
    print(f"  • Столб 1: {sensor_data[0]:.6f} с")
    print(f"  • Столб 2: {sensor_data[1]:.6f} с")
    print(f"  • Столб 3: {sensor_data[2]:.6f} с\n")
    
    # Шаг 2: Центральный ИИ рассчитывает координаты по задержкам звука
    calculated_position = locate_drone(sensor_data)
    
    # Шаг 3: Выводим результат и считаем погрешность
    print(f"[ИИ-АНАЛИТИКА] Рассчитанные координаты нарушителя:")
    print(f"  🎯 X = {calculated_position[0]:.2f} м (Ошибка: {abs(calculated_position[0]-TRUE_DRONE_POSITION[0])*100:.1f} см)")
    print(f"  🎯 Y = {calculated_position[1]:.2f} м (Ошибка: {abs(calculated_position[1]-TRUE_DRONE_POSITION[1])*100:.1f} см)")
    
    print("\n[СТАТУС] Алгоритм TDOA успешно подавил шум ветра и локализовал цель.")
