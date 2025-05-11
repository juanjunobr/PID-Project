# Manual de Usuario

**Enlace al dataset:**  
[Dataset BUSI with GT - Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)

## Requisitos del sistema

Para ejecutar el proyecto correctamente es necesario disponer de:

- **Python 3.8** o superior.
- **Sistema operativo**: Windows, Linux o macOS.
- **Hardware recomendado**:
  - CPU moderna.
  - (Opcional) GPU compatible con CUDA para acelerar el entrenamiento (recomendado).

## Instalación de dependencias

Antes de ejecutar el código, se deben instalar las librerías necesarias.  
Desde la carpeta raíz del proyecto, ejecutar:

```bash
pip install -r requirements.txt
```

Esto instalará automáticamente las dependencias necesarias:

- numpy
- opencv-python
- matplotlib
- scikit-image
- scipy
- torch
- monai

**Nota**: Si se dispone de una GPU y se desea aprovechar, se recomienda instalar `torch` con soporte CUDA siguiendo las instrucciones de https://pytorch.org/.

## Estructura del proyecto

- `main.py`: Script principal para lanzar las distintas fases del proyecto.
- `preprocessing.py`: Funciones de preprocesamiento de imágenes.
- `train_unet_monai.py`: Entrenamiento del modelo U-Net.
- `predict_with_unet.py`: Predicción de máscaras sobre nuevas imágenes.
- `metrics_evaluation.py`: Evaluación de las predicciones mediante métricas.
- `unet_busi.pt`: Archivo que guarda el modelo U-Net entrenado.

## Cómo utilizar el proyecto

### Preprocesamiento de imágenes

Para aplicar los filtros y generar imágenes procesadas, ejecutar:

```bash
python main.py classic
```

Esto aplicará:

- Filtro de Butterworth.
- Filtro de mediana.
- Extracción de características de primer orden (media y entropía).
- Extracción de características de segundo orden (bordes y cambios locales).

Los resultados se guardarán en la carpeta de salida especificada.

### Entrenamiento del modelo U-Net

Para entrenar el modelo desde cero, ejecutar:

```bash
python main.py train
```

Este comando:

- Preparará el conjunto de datos (imágenes y máscaras).
- Definirá y entrenará el modelo U-Net.
- Guardará el modelo entrenado en el archivo `unet_busi.pt`.

### Predicción con el modelo entrenado

Para realizar predicciones sobre nuevas imágenes:

```bash
python main.py predict
```

Esto:

- Cargará el modelo entrenado.
- Aplicará la predicción a nuevas imágenes.
- Guardará las máscaras predichas y comparativas visuales.

### Evaluación de resultados

Para evaluar la calidad de las predicciones:

```bash
python main.py evaluate
```

Se calcularán automáticamente las métricas:

- Dice Coefficient (DSC)
- Intersection over Union (IoU)
- Accuracy
- Precision
- Recall (Sensibilidad)
- Specificity

## Parámetros configurables

El proyecto puede ajustarse mediante la modificación de ciertos parámetros definidos en el código. Estos son los parámetros más relevantes, su función, valor por defecto, ubicación en el código y consideraciones en caso de que se desee modificarlos:

- **Número de épocas de entrenamiento (`EPOCHS`)**  
  - Archivo: `train_unet_monai.py`  
  - Ubicación: línea que contiene `EPOCHS = 30`  
  - Define el número total de pasadas completas del conjunto de datos por la red neuronal durante el entrenamiento.

- **Tamaño del batch (`BATCH_SIZE`)**  
  - Archivo: `train_unet_monai.py`  
  - Cantidad de imágenes procesadas simultáneamente antes de actualizar los pesos del modelo.

- **Tamaño de imagen (`IMG_SIZE`)**  
  - Archivos: `train_unet_monai.py`, `predict_with_unet.py`  
  - Define las dimensiones a las que se redimensionan las imágenes.

- **Tipo de preprocesamiento (`PREPROC_TAG`)**  
  - Archivos: `train_unet_monai.py`, `predict_with_unet.py`, `metrics_evaluation.py`  
  - Define el sufijo que indica el tipo de preprocesado usado en los nombres de archivo.

- **Rutas de entrada y salida**  
  - Archivo: `main.py`  
  - Parámetros: `test_dir`, `output_dir`, `gt_dir_root`, `pred_dir`

- **Modo de ejecución (`mode`)**  
  - Archivo: `main.py`  
  - Puede tomar los valores:
    - `"classic"`: aplicar preprocesamiento a todas las imágenes.
    - `"train"`: entrenar el modelo.
    - `"predict"`: realizar predicciones.
    - `"evaluate"`: calcular métricas.

**Recomendación**: antes de modificar cualquier parámetro, hacer copia de seguridad del archivo correspondiente y mantener la coherencia de rutas y nombres.

