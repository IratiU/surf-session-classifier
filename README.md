# surf-session-classifier


Proyecto de clasificación binaria para predecir si una sesión de surf será buena o no a partir de variables de previsión relacionadas con el estado del mar, el viento y el spot.
Como todos los spots tienen la misma orientación, a todos les beneficia por igual el viento o la dirección del swell por lo que a la hora de hacer el modelo, no se ha tenido en cuenta la distinción entre las playas. 


## Estructura del proyecto

surf-session-prediction/ 
│
├── data/ 
│ └── sesiones_surf.csv 
├── figures/ 
├── notebooks/ 
│   └── 01_EDA.ipynb 
│   └── 02_train.ipynb 
├── src/ 
│   ├── models.py
│   ├── evaluation.py 
│   ├── preprocessing.py 
│   ├── train.py 
│   └── predict.py 
├── models/ 
│   
├── requirements.txt 
└── README.md




##  Datos

El dataset contiene información de sesiones de surf en distintos spots. La variable objetivo es `buena_sesion`, que indica si la sesión fue buena (`Yes`) o no (`No`).

Durante la exploración inicial se detectaron varios aspectos relevantes:

Existen duplicados en el dataset.
La variable fecha está almacenada como texto y se transforma a formato fecha.
Algunas variables numéricas contienen valores faltantes o están almacenadas como texto.
La variable altura_ola_m contiene algunos valores erróneos iguales a 99.
La variable spot presenta diferencias de formato, como mayúsculas, minúsculas o espacios.
La variable objetivo está desbalanceada.
Algunas variables, como id_sesion, no aportan información útil al modelo.
La variable num_surfistas no se utiliza porque no estaría disponible antes de que ocurra la sesión.

## Preprocesado

El preprocesado se ha organizado en un transformer compatible con scikit-learn, de forma que pueda integrarse dentro de un Pipeline y aplicarse de manera consistente tanto en entrenamiento como en inferencia.

Las principales transformaciones realizadas son:

* Eliminación de duplicados en el dataset de entrenamiento.
* Eliminación de filas con valores claramente erróneos en altura_ola_m.
* Conversión de fecha a formato datetime y extracción del mes.
* Normalización de los nombres de los spots.
* Limpieza de variables categóricas como marea y direccion_viento.
* Conversión de variables numéricas a formato numérico.
* Imputación de temp_agua usando la media por spot y mes.
* Imputación de viento_kmh usando la mediana por spot y mes.
* Eliminación de columnas que no deben entrar al modelo, como fecha, id_sesion y num_surfistas.


La variable valoracion se mantiene porque se interpreta como una variable disponible en la previsión previa a la sesión.
No se hace un escalado de las variables ya que al hacer un Random Forest, no es necesario. 

## Entrenamiento

Para entrenar los modelos:

python src/train.py

Para ejecutar el entrenamiento sin hacer un GridSearchCV

python src/train.py --no-tune

Para indicar otro ruta de datos:

python src/train.py --data data/sesiones_surf.csv

El script entrena:

- un modelo baseline con DummyClassifier;
- un modelo Random Forest inicial;
- opcionalmente, un Random Forest ajustado con GridSearchCV y validación temporal.

Los modelos se guardan automáticamente en la carpeta models/.


## Evaluación

La división entre train y test se realiza respetando el orden temporal de las sesiones. Aunque el modelo Random Forest utiliza remuestreo bootstrap internamente, este remuestreo se aplica únicamente sobre los datos de entrenamiento, por lo que el conjunto de test sigue representando datos no vistos.

Esta estrategia busca aproximar mejor el caso real de uso: entrenar con sesiones pasadas y predecir sesiones futuras. No obstante, al tratarse de datos ficticios, esta decisión no es una necesidad estricta.
Las métricas de validación incluyen

* Accuracy: Por ser la métrica estándar, aunque debido al desbalanceo de las clases, no es la mejor métrica de validación. 

* Precision: Cuantas predicciones buenas lo son realmente.

* Recall: Tasa de verdaderos positivos. Cuantas sesiones buenas detecta realmente.

* F1-score: La media armónica entre precision y recall.

* Curva ROC y AUC-ROC: permiten evaluar la capacidad del modelo para separar sesiones buenas y malas a distintos umbrales de decisión. Esto es interesante porque, en una aplicación real, se podría ajustar el umbral de probabilidad según se quiera recomendar más sesiones o recomendar solo aquellas con mayor confianza.

Se presta especial atención a F1-score porque combina precision y recall. En este problema tanto el `Yes` como el `No` tienen relevancia. 

## Predicción sobre nuevas sesiones

Una vez entrenado y guarado el modelo, se puede usar src/predict.py para predecir nuevas sesiones de la siguiente forma:

python src/predict.py --input data/nuevas_sesiones.csv


Para usar un modelo concreto:

python src/predict.py \
  --input data/nuevas_sesiones.csv \
  --model models/random_forest_tuned.pkl \
  --output data/predicciones.csv


El archivo de entrada debe contener las mismas columnas predictoras que el dataset original.

El archivo de salida incluye la predicción (Yes / No) y, si el modelo lo permite, la probabilidad estimada de buena sesión.

## Resultados y conclusiones

El baseline sirve como referencia mínima para comprobar si los modelos de ML aportan valor real a la predicción o un modelo que predice siempre que no es mejor. 

El Random Forest permite capturar relaciones no lineales entre las condiciones de mar, viento y spot. Además, al estar integrado en un pipeline completo, el modelo entrenado puede aplicarse directamente a nuevas sesiones sin repetir manualmente el preprocesado. 

La solución es razonable para una primera versión del problema, aunque tiene algunas limitaciones:

- algunas variables, como la dirección del viento, la marea o el swell, pueden tener efectos distintos según el spot.
- el modelo global puede no capturar perfectamente las particularidades de cada playa.
- no se ha probado una validación específica por spot.

Como siguientes pasos, probaría:

- Modelos específicos por spot si hubiese suficientes datos.
- Calibrar las probabilidades.
- Comparar con modelos como Gradient Boosting o XGBoost 



Los resultados de los métodos son los siguientes:




[INFO] Distribucion de clases:
              train   test
buena_sesion              
No            0.718  0.729
Yes           0.282  0.271

===== Baseline - Train =====
Accuracy:          0.718
Balanced accuracy: 0.500
Precision Yes:     0.000
Recall Yes:        0.000
F1 Yes:            0.000
ROC AUC:           0.500

Classification report:
              precision    recall  f1-score   support

          No       0.72      1.00      0.84      1143
         Yes       0.00      0.00      0.00       450

    accuracy                           0.72      1593
   macro avg       0.36      0.50      0.42      1593
weighted avg       0.51      0.72      0.60      1593



===== Baseline - Test =====
Accuracy:          0.729
Balanced accuracy: 0.500
Precision Yes:     0.000
Recall Yes:        0.000
F1 Yes:            0.000
ROC AUC:           0.500

Classification report:
              precision    recall  f1-score   support

          No       0.73      1.00      0.84       291
         Yes       0.00      0.00      0.00       108

    accuracy                           0.73       399
   macro avg       0.36      0.50      0.42       399
weighted avg       0.53      0.73      0.62       399

[INFO] Modelo guardado en: /home/irati/surf-session/surf-session-classifier/models/baseline.pkl

============================================================
RANDOM FOREST  (parametros iniciales)
============================================================
===== Random Forest - Train =====
Accuracy:          0.812
Balanced accuracy: 0.837
Precision Yes:     0.616
Recall Yes:        0.893
F1 Yes:            0.729
ROC AUC:           0.934

Classification report:
              precision    recall  f1-score   support

          No       0.95      0.78      0.86      1143
         Yes       0.62      0.89      0.73       450

    accuracy                           0.81      1593
   macro avg       0.78      0.84      0.79      1593
weighted avg       0.85      0.81      0.82      1593



===== Random Forest - Test =====
Accuracy:          0.764
Balanced accuracy: 0.795
Precision Yes:     0.541
Recall Yes:        0.861
F1 Yes:            0.664
ROC AUC:           0.895

Classification report:
              precision    recall  f1-score   support

          No       0.93      0.73      0.82       291
         Yes       0.54      0.86      0.66       108

    accuracy                           0.76       399
   macro avg       0.74      0.79      0.74       399
weighted avg       0.83      0.76      0.78       399

[INFO] Matriz de confusion guardada en: /home/irati/surf-session/surf-session-classifier/figures/confusion_matrix_Random_Forest.png
[INFO] Modelo guardado en: /home/irati/surf-session/surf-session-classifier/models/random_forest.pkl

============================================================
RANDOM FOREST TUNING  (GridSearchCV + TimeSeriesSplit)
============================================================
[INFO] Iniciando GridSearchCV (puede tardar varios minutos)...
Fitting 3 folds for each of 18 candidates, totalling 54 fits
[INFO] Mejores hiperparametros: {'classifier__class_weight': 'balanced', 'classifier__max_depth': None, 'classifier__min_samples_leaf': 5, 'classifier__n_estimators': 300}
[INFO] Mejor F1 medio en validacion temporal: 0.719

===== RF tuned - Train =====
Accuracy:          0.907
Balanced accuracy: 0.922
Precision Yes:     0.771
Recall Yes:        0.956
F1 Yes:            0.853
ROC AUC:           0.982

Classification report:
              precision    recall  f1-score   support

          No       0.98      0.89      0.93      1143
         Yes       0.77      0.96      0.85       450

    accuracy                           0.91      1593
   macro avg       0.88      0.92      0.89      1593
weighted avg       0.92      0.91      0.91      1593



===== RF tuned - Test =====
Accuracy:          0.810
Balanced accuracy: 0.814
Precision Yes:     0.610
Recall Yes:        0.824
F1 Yes:            0.701
ROC AUC:           0.904

Classification report:
              precision    recall  f1-score   support

          No       0.92      0.80      0.86       291
         Yes       0.61      0.82      0.70       108

    accuracy                           0.81       399
   macro avg       0.77      0.81      0.78       399
weighted avg       0.84      0.81      0.82       399

[INFO] Matriz de confusion guardada en: /home/irati/surf-session/surf-session-classifier/figures/confusion_matrix_RF_tuned.png
[INFO] Modelo guardado en: /home/irati/surf-session/surf-session-classifier/models/random_forest_tuned.pkl

============================================================
COMPARATIVA EN TEST
============================================================
               accuracy  balanced_accuracy  precision_yes recall_yes  f1_yes  roc_auc
Baseline          0.729              0.500          0.000       0.000   0.000    0.500
Random Forest     0.764              0.795          0.541       0.861   0.664    0.895
RF tuned          0.810              0.814          0.610       0.824   0.701    0.904


Viendo las comparaciones de las métricas de validación, vemos que el RF aporta valor respecto al baseline y que en general, el GridSearch aún no siendo muy amplio, si que mejora algo el rendimiento. 

El recall= 0.861 significa que detecta aproximadamente el 86.1% de las sesiones buenas reales, pero al tener una precisión de 0.541 significa que mete bastantes falsos positivos. 

En el caso del RF ajustado, el precisión sube aunque el recall baja. Esto quiere decir que detecta algo menos de sesiones buenas pero cuando la detecta hay más probabilidad de que esa sesión sea realmente buena. 
El F1 score sube lo que significa que hay un mejor equilibrio entre recall y precision. 








