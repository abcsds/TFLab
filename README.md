# Tensorflow lab

use docker image to setup environment:

 - Install cuda-docker
 - Build Dockerfile: `docker build -t tf:latest .`
 - Run container: `docker run --runtime=nvidia -v $PWD:/tmp --net=host tf`

virtualenv:

`mkvirtualenv [-a project_path] [-i package] [-r requirements_file] [virtualenv options] ENVNAME`

mkvirtualenv -a ./ -r requirements.txt TF
