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
| ExtraTrees | 48.6 % | 43.5 % |
| RandomForest | 48.77 % | 43.8 % |

Se pueden consultar más detalles en el archivo doc/Documentación.pdf

# Instalación y ejecución

- pip install -r requirements

#### EDAs

Un modo cómodo de ver los datos y sacar algunas conclusiones son los EDAs. Puedes verlos en tu navegador mediante archivos html o con jupyter si deseas modificarlos. Están en la carpeta edas.

#### Preprocesado

- cd preprocessing
- python preprocessing.py

#### Ejecución de diferentes algoritmos

- cd ../models/
- python et_ensemble.py
- python rf_ensemble.py

# TODO

Estudiar con más detalle las secuencias de compras que muestra los clientes, aunque esto se recoge en el preprocesado llevado a cabo, creemos que con Redes Neuronales Recurrentes (RNN, LSTM o GRU) podemos obtener mejores resultados.

# Notas

Para futuras competiciones recomendamos al grupo Cajamar que valore de modo diferentes los resultados, dando más importancia al porcentaje de aciertos. Nos hemos visto superados en la primera fase de Granada por un equipo que en la fase final ha logrado un 45% de acierto y en la clasificación general por uno con 46% de acierto.
