FROM python:3.9

WORKDIR /app

COPY . .

CMD ['step1.sh']

CMD ['python3 step5.py']

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
