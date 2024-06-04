FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install Flask kiwipiepy langdetect -y

CMD ['step1.sh']

CMD ['python3 step5.py']

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
