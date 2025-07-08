# Usa una imagen oficial de Python 3.10
FROM python:3.10-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia primero el requirements.txt e instala dependencias
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Ahora copia el resto de archivos del proyecto
COPY . .

# Expone el puerto donde se ejecutará la API
EXPOSE 8000

# Comando para arrancar FastAPI con uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
