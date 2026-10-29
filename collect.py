import subprocess
import itertools
import time
import os

# 1. Задайте списки параметров для перебора (grid search)
epochs_list = [10000, 25000, 50000, 100000]
learning_rates = [0.1, 0.05, 0.01, 0.001]
layers_list = ["2,32,1", "2,64,1", "2,128,1", "2,64,32,1"]

# Программа для запуска
executable = "./Neuro.Perceptron/bin/Debug/net10.0/linux-x64/Neuro.Perceptron"

# Создаем все возможные комбинации параметров
combinations = list(itertools.product(epochs_list, learning_rates, layers_list))
total_runs = len(combinations)

print(f"Всего комбинаций для запуска: {total_runs}")
print("-" * 40)

start_time = time.time()

# 2. Запускаем цикл по всем комбинациям
for index, (epoch, lr, layers) in enumerate(combinations, start=1):
    # Формируем команду запуска
    command = [
        executable,
        "--epochs", str(epoch),
        "--learningRate", str(lr),
        "--layers", layers,
        "-f"
    ]
    
    cmd_str = " ".join(command)
    print(f"[{index}/{total_runs}] Запуск: {cmd_str}")
    
    try:
        # Выполняем команду
        # stdout=subprocess.DEVNULL скроет стандартный вывод программы, если он мешает. 
        # Если хотите видеть логи нейросети, уберите этот параметр.
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении: {cmd_str}")
        print(f"Код ошибки: {e.returncode}")
    except FileNotFoundError:
        print(f"Ошибка: Исполняемый файл '{executable}' не найден.")
        break

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print("-" * 40)
print(f"Готово! Все варианты отработали за {elapsed_minutes:.2f} минут.")
