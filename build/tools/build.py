#
# build.py - Компилирует вашу программу в бинарный файл при помощи pyinstaller.
#
# log-level: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
#


# Импортируем:
import os
import json
import time
import shutil
from threading import Thread


# Переменные (НЕ НАСТРАИВАЕМЫЕ):
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
wait_active   = True
wait_text     = "> wait... "
wait_text_len = len(wait_text)+1


# Ожидание:
def waiting() -> None:
    symbols = r"/─\|"
    while wait_active:
        for s in symbols:
            if not wait_active: continue
            print("\r"+wait_text+s, end="")
            time.sleep(0.1)


# Очищаем консоль:
def clear_console() -> None:
    if os.name == "nt": os.system("cls")
    else: os.system("clear")


# Основная функция:
def main() -> None:
    global wait_active, wait_text_len

    clear_console()  # Очищаем консоль.

    # Читаем конфигурационный файл сборки:
    with open("../config.json", "r+", encoding="utf-8") as f: config = json.load(f)

    # Преобразование данных конфигурации в переменные:
    main_file         = config["main-file"]
    program_icon      = config["program-icon"]
    program_name      = config["program-name"]
    console_disabled  = config["console-disabled"]
    data_folder       = config["data-folder"]
    pyinstaller_flags = config["pyinstaller-flags"]
    lg                = "--log-level "
    waiting_enabled   = any(flag in pyinstaller_flags for flag in [lg+"WARN", lg+"ERROR", lg+"FATAL"])

    # Отдельный поток для вывода ожидания:
    waiting_thread = Thread(target=waiting, daemon=True)
    if not waiting_enabled: wait_text_len = 0

    # Генерация флагов компиляции:
    flags = f"--noconfirm -n=\"{program_name}\" "
    for flag in pyinstaller_flags: flags += f"{flag} "
    if console_disabled:           flags +=  "--noconsole "
    if program_icon is not None:   flags += f"--icon=../../{program_icon} "

    # Собираем проект:
    print(f"{' COMPILATION PROJECT ':─^96}\n")
    if waiting_enabled: waiting_thread.start()

    os.system(f"pyinstaller {flags} ../../{main_file}")

    print(f"\r{' '*wait_text_len}\n> COMPILATION IS SUCCESSFUL!\n\n{'─'*96}\n\n")

    # Удаляем мусор и собираем всё в одну папку:
    print("Deleting temporary build files...")

    # Удаляем прочие лишние файлы и папки:
    if os.path.isdir("../out/"): shutil.rmtree("../out/")
    if os.path.isdir("build"): shutil.rmtree("build")
    for file in os.listdir():
        if file.endswith(".spec"): os.remove(file)

    # Копируем папку с данными:
    print("\n\n\nCopying the data folder...")

    # Копируем содержимое сборки и содержимое data папки в папку out:
    if os.path.isdir("./dist/"):
        shutil.copytree("./dist/", "../out/", dirs_exist_ok=True)
        shutil.rmtree("./dist/")
        shutil.copytree(f"../../{data_folder}", f"../out/{os.path.basename(os.path.normpath(data_folder))}")

    wait_active = False
    print("\rDone!"+' '*wait_text_len)

    print("\n\nOutput of build in folder: /build/out/")


# Если этот скрипт запускают:
if __name__ == "__main__":
    main()
