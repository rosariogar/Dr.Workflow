# Usa una imagen base de Python
FROM python:3.8

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos necesarios a la imagen
COPY main.py .
COPY componentes.py .
COPY requirements.txt .

# Instala las dependencias
RUN pip install -r requirements.txt

# Comando para ejecutar tu programa
CMD ["python", "main.py"]