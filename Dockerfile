# slim Python image 
FROM python:3.12-slim

# working directory 
WORKDIR /app

COPY requirements.txt .

# installing dependencies 
RUN pip install --no-cache-dir -r requirements.txt

COPY . . 

RUN mkdir  -p models 

EXPOSE 8000

CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]

