# Usa la imagen base de Python
FROM python:3.8

# Configura el directorio de trabajo en el contenedor
WORKDIR /app

# Copia los archivos locales al contenedor
COPY main.py /app
COPY componentes.py /app
COPY requirements.txt /app
COPY data /app/data

# Crea la carpeta resultados y copia los archivos locales al contenedor
COPY resultados /app/resultados

# Instala las dependencias
RUN pip install -r requirements.txt

# Ejecuta el archivo main.py cuando se inicie el contenedor
CMD ["python", "main.py"]
