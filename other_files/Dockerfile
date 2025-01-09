# Use AWS Lambda Python 3.9 base image
FROM public.ecr.aws/lambda/python:3.9

# Install system dependencies required for psycopg and other libraries
RUN yum install -y gcc libpq-devel python3-devel

# Set the working directory inside the container
WORKDIR /var/task

# Copy application files
COPY app.py session_manager.py pinecone_index.py pinecone_db.py ./
COPY requirements.txt ./
COPY .env ./

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Set the CMD to point to your Lambda handler
CMD ["app.lambda_handler"]
