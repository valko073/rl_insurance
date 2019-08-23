FROM python:3.7
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD python ./run_insurance.py --comet --num_agents=3 --num_insurances=3 --num_steps=100000
