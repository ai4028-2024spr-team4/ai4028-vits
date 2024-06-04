FROM python:3.9

WORKDIR /app

COPY . .

RUN pip install Flask kiwipiepy langdetect

CMD ['step1.sh']

CMD ['python3 step5.py']

EXPOSE 5000

CMD ["flask", "--app", "app", "run"]
