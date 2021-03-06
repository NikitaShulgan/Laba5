# Laba5
## Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
### 2. С использованием примера [2], техники обучения Transfer Learning [1], оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife

#### owl-1617721605.0678256 ``` train_4 ```

[Tensorboard](https://tensorboard.dev/experiment/1ku6PaQQRVOrNOR1ksrgvA/#scalars&runSelectionState=eyJ0cmFpbiI6ZmFsc2V9&_smoothingWeight=0)

![image](https://user-images.githubusercontent.com/80168174/113783107-eec9cf80-973b-11eb-93cf-b5d1fcee2be4.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/2_epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/2_epoch_loss.svg">


### 3. С использованием техники обучения Fine Tuning дополнительно обучить нейронную сеть EfficientNet-B0 предварительно обученную в пункте 2.

[Tensorboard](https://tensorboard.dev/experiment/nIvrrs8HQje3UDYC4FbnAw/#scalars&runSelectionState=eyJvd2wtMTYxNzczNjcwMi4wNjIxNzIvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzczNjcwMi4wNjIxNzIvdmFsaWRhdGlvbiI6ZmFsc2UsIm93bC0xNjE3NzM3NTQ3LjM3ODYzMzUvdHJhaW4iOmZhbHNlLCJvd2wtMTYxNzczNzU0Ny4zNzg2MzM1L3ZhbGlkYXRpb24iOmZhbHNlLCJvd2wtMTYxNzczODYxMy44ODI2MDI1L3RyYWluIjpmYWxzZSwib3dsLTE2MTc3Mzg2MTMuODgyNjAyNS92YWxpZGF0aW9uIjpmYWxzZSwib3dsLTE2MTc3NDA0NTkuMjk1NjM1NS90cmFpbiI6ZmFsc2UsIm93bC0xNjE3NzQwNDU5LjI5NTYzNTUvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTc3NDIyNTAuNzM2NzUzMi90cmFpbiI6ZmFsc2UsIm93bC0xNjE3NzQyMjUwLjczNjc1MzIvdmFsaWRhdGlvbiI6dHJ1ZSwib3dsLTE2MTc3NDQwNzMuODk2Nzg3L3RyYWluIjpmYWxzZSwib3dsLTE2MTc3NDQwNzMuODk2Nzg3L3ZhbGlkYXRpb24iOnRydWV9&_smoothingWeight=0)

#### owl-1617740459.2956355 ``` lr = 0.0000001 ```

#### owl-1617742250.7367532 ``` lr = 0.00000001 ```

#### owl-1617744073.896787 ``` lr = 0.00000005 ```

![image](https://github.com/NikitaShulgan/Laba5/blob/main/For_Readme/image_2021-04-07_00-53-21.png)

#### epoch_categorical_accuracy
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/epoch_categorical_accuracy.svg">

#### epoch_loss
<img src="https://raw.githubusercontent.com/NikitaShulgan/Laba5/main/For_Readme/epoch_loss.svg">

### Вывод 
#### Используя технику обучения Fine Tuning с ``` lr = 1e-7 ``` удалось улучшить результат на 0.35%.
