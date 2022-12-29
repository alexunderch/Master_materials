FROM python:3.9.15-bullseye

# We don't want to run our application as root if it is not strictly necessary, even in a container.
# Create a user and a group called 'app' to run the processes.
# A system user is sufficient and we do not need a home.
RUN apt-get update
RUN apt-get install -y unzip git wget redis-server gunicorn
                
RUN adduser --system --group --no-create-home app

# Place the application components in a dir below the root dir

COPY . /app

EXPOSE 5555 6969
# Make the directory the working directory for subsequent commands
WORKDIR /app

RUN chmod +x prepare_data.sh
RUN ./prepare_data.sh

# Install from the requirements.txt we copied above
RUN pip install -r requirements.txt


# Hand everything over to the 'app' user
RUN chown -R app:app /app

# Subsequent commands, either in this Dockerfile or in a
# docker-compose.yml, will run as user 'app'
# USER app

# We are done with setting up the image.
# As this image is used for different
# purposes and processes no CMD or ENTRYPOINT is specified here,
# this is done in docker-compose.yml.
# CMD ["python", "server/main.py"]
