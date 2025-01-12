#!/usr/bin/env bash

if ! command -v jq &>/dev/null; then
    echo "jq не найден. Пожалуйста, установите jq и попробуйте снова."
    exit 1
fi


# Функция для обработки одного файла .ipynb
process_file() {
    local filename="$1"

    # Применяем jq для удаления секции .metadata.widgets
    jq -M 'del(.metadata.widgets)' "$filename" > temp_ipynb && mv temp_ipynb "$filename"

    # Вызываем Python-скрипт для применения функции set_cell_id - обновление идентификаторов ячейки
    python3 Tools/notebook.py "$filename"
}

# экспортируйте функцию, чтобы она была известна в дочерних процессах
export -f process_file

# Рекурсивный обход всех файлов с расширением .ipynb
find . -type f -name "*.ipynb" -exec bash -c 'process_file "$0"' {} \;

# Удаление временного файла, если он существует
if [ -f temp_ipynb ]; then
    rm temp_ipynb
fi

echo "Обработка завершена."
