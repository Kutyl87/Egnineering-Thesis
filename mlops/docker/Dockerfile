FROM python:3.11
WORKDIR /usr/src/app
COPY ../requirements_mlflow.txt ./
RUN ls
RUN pip install --no-cache-dir -r requirements_mlflow.txt
CMD mlflow server --host ${HOSTNAME} --port ${PORT}
