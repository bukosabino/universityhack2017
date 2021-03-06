# Cajamar UniversityHack 2017 Datathon

Es la competición online de analítica en base a datos bancarios reales del Grupo Cajamar orientado a los mejores centros formativos en Data Science.

http://www.cajamardatalab.com/datathon-cajamar-universityhack-2017/

# Planteamiento y Resultados

A partir de los datos históricos de productos contratados por un cliente, se pretende adivinar cuál será el siguiente producto a contratar por un cliente.

Hemos planteado este problema como un problema de clasificación, multiclase, en la que usaremos como target el próximo producto a comprar por el usuario y como variables independientes las 5 variables sociodemográficas y las 94 columnas binarias resultantes de la binarización de los 94 posibles productos.

Los datos que usamos para validación serán la última compra de cada uno de los clientes, dejando en el train los primeros productos adquiridos por el cliente. En el caso de que el cliente solo tenga 1 producto con la entidad lo descartamos porque no tendremos modo de validar nuestra predicción.

Los algoritmos usados son algunos de los ensembles más famosos que nos provee la librería scikit-learn en python.

| Algoritmo | Precisión | Coeficiente Kappa |
| --------- | --------- | ----------------- |
| ExtraTrees (models/et_ensemble.py) | 48.6 % | 43.5 % |
| RandomForest (models/rf_ensemble.py) | 48.77 % | 43.8 % |
| XGBoost (models/xgb_ensemble.py) | 50.88 % | 46.06 % |

Se pueden consultar más detalles en el archivo doc/Documentación.pdf

# Instalación y ejecución

- pip install -r requirements

Si se quiere usar el archivo xgb_ensemble.py donde se usa el XGBoost, es necesario no instalar con un simple “pip install xgboost”, porque se han usado características recientemente incluidas en el proyecto. Para más info: https://github.com/dmlc/xgboost/issues/1950 . Por tanto, los pasos a seguir para instalar XGBoost serían:

- git clone https://github.com/dmlc/xgboost
- cd xgboost/python-package
- sudo python setup.py install
- export PYTHONPATH=~/<your-path>/xgboost/python-package

#### EDAs

Un modo cómodo de ver los datos y sacar algunas conclusiones son los EDAs. Puedes verlos en tu navegador mediante archivos html o con jupyter si deseas modificarlos. Están en la carpeta edas.

#### Preprocesado

- cd preprocessing
- python preprocessing.py

#### Ejecución de diferentes algoritmos

- cd ../models/
- python et_ensemble.py
- python rf_ensemble.py
- python xgb_ensemble.py

# TODO

Estudiar con más detalle las secuencias de compras que muestra los clientes, aunque esto se recoge en el preprocesado llevado a cabo, creemos que con Redes Neuronales Recurrentes (RNN, LSTM o GRU) podemos obtener mejores resultados.

# Notas

Para futuras competiciones recomendamos al grupo Cajamar que valore de modo diferente los resultados obtenidos por los participantes, dando más importancia al porcentaje de aciertos. Nos hemos visto superados en la primera fase de Granada por un equipo que en la fase final ha logrado un 45% de acierto y en la clasificación general ha ganado un equipo con un 46% de acierto.

Les proponemos usar plataformas como Kaggle, porque pueden facilitar la labor en la valoración de resultados, la liberación de los datos de la competición (tanto en modo privado para los participantes, como hacerlos públicos si así lo desean), además aumentaría la comunicación entre participante mediante foros comunes.
