FROM tensorflow/tensorflow:2.16.1-gpu

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_USE_LEGACY_KERAS=1

RUN apt-get update && apt-get install -y --no-install-recommends git && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY ./src ./src

RUN chmod +x src/run.sh
CMD ["bash", "src/run.sh"]