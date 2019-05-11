FROM tensorflow/tensorflow:latest-gpu-py3-jupyter

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install sklearn matplotlib

# Make port 80 available to the world outside this container
# EXPOSE 8888

# Define environment variable
# ENV NAME World

# RUN mkdir /root/tf/project
WORKDIR /tmp

# Run jupyter when the container launches
CMD ["jupyter", "notebook", "--allow-root", "--no-browser"]

# ENTRYPOINT ["jupyter", "notebook", "--allow-root", "--no-browser"]

# docker run --runtime=nvidia -it --rm -v $PWD:/tmp -w /tmp --net=host tensorflow/tensorflow:latest-gpu-py3-jupyter bash


# docker run --runtime=nvidia -it --rm -v $PWD:/tmp -w /tmp --net=host  --entrypoint "jupyter notebook --allow-root --no-browser" tensorflow/tensorflow:latest-gpu-py3-jupyter


# docker run --runtime=nvidia -v $PWD:/tmp --net=host tf
