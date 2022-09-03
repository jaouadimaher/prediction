FROM python:3.9.7
EXPOSE port 8501
CMD [ "executable" ] mkdir -p /app
WORKDIR /app
COPY source dest requirements.txt ./requirements.txt
RUN command pip install -r requirements.txt
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["test.py"]