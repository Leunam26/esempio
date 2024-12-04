# Usa un'immagine di base con Python
FROM python:3.9-slim

# Installa MLflow
RUN pip install mlflow[extras]

# Espone la porta 5000 (usata da MLflow)
EXPOSE 5000

# Comando per avviare il server MLflow
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
