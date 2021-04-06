# Laba5
## Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
### 2. С использованием примера [2], техники обучения Transfer Learning [1], оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife

#### owl-1617721605.0678256

![image](https://user-images.githubusercontent.com/80168174/113783107-eec9cf80-973b-11eb-93cf-b5d1fcee2be4.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/2_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/2_epoch_loss.svg">


### 3. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 2.

#### owl-1617740459.2956355 ``` lr = 0.0000001 ```

#### owl-1617742250.7367532 ``` lr = 0.00000001 ```

#### owl-1617744073.896787 ``` lr = 0.00000005 ```

![image](https://github.com/NikitaShulgan/Laba5/blob/main/For_Readme/image_2021-04-07_00-53-21.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/epoch_loss.svg">
