FROM continuumio/miniconda3:4.12.0

WORKDIR /app

COPY requirements.txt .

RUN conda create -n pictora python=3.9 -y && \
    conda run -n pictora pip install --no-cache-dir -r requirements.txt && \
    conda clean --all -f -y

ENV CUDA_VISIBLE_DEVICES=""
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONPATH=/app

COPY . .

EXPOSE 8000

CMD ["conda", "run", "--no-capture-output", "-n", "pictora", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
