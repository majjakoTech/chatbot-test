FROM python:3.12.7
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 5000
CMD ["uvicorn","main:app","--host","0.0.0.0","--port","5000","--reload"]