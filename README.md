# Dr. Workflow - Análisis de Datos para Profesionales de la Salud

Dr. Workflow es un flujo de trabajo semi-automatizado diseñado para ayudar a los profesionales de la salud en el análisis eficiente de datos de pacientes, sin importar la patología en cuestión.

## Introducción

En el ámbito de la atención médica, el análisis de datos es crucial para tomar decisiones informadas sobre el tratamiento de los pacientes. Dr. Workflow simplifica y agiliza este proceso al proporcionar herramientas automatizadas y guías paso a paso para el análisis de datos. Ya sea que trabajes en investigación clínica, epidemiología o cualquier otro campo relacionado, Dr. Workflow te ayudará a obtener información valiosa de tus datos.

## Características Principales

- Flujo de trabajo semi-automatizado para el análisis de datos.
- Permite personalizar flujos de trabajo según las necesidades específicas.
- Facilita la carga, limpieza y visualización de datos.
- Admite múltiples tipos de patologías y conjuntos de datos.
- Proporciona resultados y visualizaciones claras y comprensibles.

## Instalación
1. Abre una terminal y navega a la carpeta en tu sistema donde deseas clonar este repositorio. Por ejemplo:
- En Windows:
   ```bash
   cd C:\ruta\deseada\
   
- En Linux/Mac:
   ```bash
   cd /ruta/deseada/
  
2. Clona este repositorio en tu máquina local:
   ```bash
   git clone https://github.com/rosariogar/Dr.WorkFlow.git
   
3. Navega a la carpeta del proyecto:
   ```bash
   cd DrWorkFlow
   
5. Construye la imagen Docker ejecutando el siguiente comando:
   ```bash
   docker build -t dr_workflow .

6. Ejecuta el contenedor y accede a la interfaz de línea de comandos:
   ```bash
     docker run -it dr_workflow
7. Sigue las instrucciones en la línea de comandos para cargar datos y realizar análisis.

### Requisitos Previos

- Docker: Asegúrate de tener Docker instalado en tu sistema antes de comenzar. Puedes descargarlo [aquí](https://www.docker.com/products/docker-desktop/).

## Uso de Datos de Ejemplo
Si deseas probar el proyecto con datos reales, puedes descargar el archivo CSV de ejemplo [aquí](https://www.kaggle.com/datasets/emmanuelfwerr/thyroid-disease-data). Sigue las instrucciones en la interfaz de línea de comandos para cargar y utilizar el archivo CSV de ejemplo.

