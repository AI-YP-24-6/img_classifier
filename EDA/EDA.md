# Разведочный анализ (EDA) датасета изображений овощей и фруктов
Для разведочного анализа используется подготовленный датасет, состоящий из изображений овощей и фруктов.
Разведочный анализ каждого датасета по отдельности:
- [EDA Fruits360](/EDA/EDA_Fruits360.md)
- [EDA Vegetables](/EDA/EDA_Vegetables.md)
- [EDA tasty_fruit](/EDA/EDA_tasty_fruit.md)

[Ноутбук с подготовкой датасета](/EDA/dataset_merging.ipynb) - При подготовке датасета были объединены классы и применен oversampling путем изменения контрастности, яркости, угла и отзеркаливания изображений и undersampling путем отбрасывания случайных изображений из класса для балансировки.
**В датасете представлено 33 класса по 1400 изображений в каждом.**  
[Ноутбук с EDA датасета изображений](/EDA/EDA_main.ipynb)
## Общая информация о данных
### Средние, минимальные и максимальные размеры изображений
* Среднее значение ширины - 180px, высоты - 171px  
* Минимальные размеры для ширины и высоты - 100px. Значительная часть (около 25% датасета) представлена в таком размере.  
* Максимальное значение для ширины - 453px, высоты - 363px.

Половина изображений имеют ширину до 224px и высоту до 194px, а три четверти изображений имеют размер по ширине и высоте до 224px.
### Средние значения и отклонения по каналам (R, G, B)
Срдение значения и стандартные отклонения по каналам сильно варьируется от класса к классу. В некоторых классах имеются похожие изображения (изображения одного и того же объекта под разными углами, к примеру), это сказывается на стандартном отклонении.  
### Метаданные изображений
У 4234 изображений из 46200 присутствуют EXIF-данные.
Для анализа были взяты: `model, aperture_value, brightness_value, focal_length, digital_zoom_ratio`
При визуальном изучении стало заметно, что изображения с различным фокусным расстоянием и яркостью отличаются друг от друга по этим признакам.
Также качество изображений отличается в зависимости от модели камеры.  
## Баланс классов
В каждом классе имеется 1400 изображений. Датасет хорошо сбалансирован.
## Размер изображений
Изображения в датасете имеют разный размер. Следует привести изображения к единому размеру.  
При изменении размера изображений к 224x224px большинство изображений не потеряет деталей, также **данный размер используется в нейронных сетях ResNet, VGG**.  
При приведении датасета к данному размеру многие изображения не придется изменять. Крупных изображений немного, их можно уменьшить без сильной потери информациии.

**Решено использовать размер для изображений 224*224px**
## Выбросы
1. Представлено 2 класса клубники: Strawberry и Strawberries. Следует объеденить данные классы в один путем удаления части изображений из Strawberry и добавления части изображений из Strawberries в Strawberry (60% Strawberries и 40% Strawberry, к примеру), затем класс Strawberries удалить.  
2. Присутствуют изображения сильно крупнее большинства других и изображения размером 100*100px.  
   При изменении размера изображений:
   - Крупные не сильно повлияют на качество модели, т.к. их немного
   - Изображения 100*100px имеют низкую детализацию, объект на них может быть сложно классифицировать