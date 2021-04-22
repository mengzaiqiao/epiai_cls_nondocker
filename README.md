## Running the docker container

To run the demo you must:
- Install Docker; You can run either the following script or the [official instruction](https://docs.docker.com/engine/install/ubuntu/).
```shell
sudo apt-get update
sudo apt-get install apt-transport-https ca-certificates curl gnupg-agent software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
apt-key fingerprint 0EBFCD88
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
apt update
apt-get install docker-ce docker-ce-cli containerd.io
```
- Clone this repo;
- Open a terminal and navigate into the `docker-scripts` directory;
- Run `./build_image.sh`;
- Run `./start_container.sh -s`. 

# Document Classification Demo

This repository contains a web API demo for performing document classification for EPI-AI project. 

- Demo 1 (GET+txt), click here ->: [http://172.25.17.6:5000/cls_text?txt='Today I woke up with migraine and I took an aspirine.'](http://172.25.17.6:5000/cls_text?txt=Today%20I%20woke%20up%20with%20migraine%20and%20I%20took%20an%20aspirine.)
- Demo 2 (GET+txt), click here ->: [http://172.25.17.6:5000/cls_text?txt='Niger states have reported confirmed human cases of avian influenza H5N1, also called bird flu.' ](http://172.25.17.6:5000/cls_text?txt=Niger%20states%20have%20reported%20confirmed%20human%20cases%20of%20avian%20influenza%20H5N1,%20also%20called%20bird%20flu.)
- Demo 3 (POST+json, ), run with command line:
```shell
data='{"docs":["Today I woke up with migraine and I took an aspirine.", "Niger states have reported confirmed human cases of avian influenza H5N1, also called bird flu."]}'
url=http://172.25.17.6:5000/cls_json
curl --header "Content-Type: application/json" --request POST --data $data $url
```

### Server Performance & Security

Please be aware that this project is to be regarded as a technical demo only and it not meant to be production ready. The server
uses Flask's development server; to enhance performance and security, you should choose one of Flask's 
[deployment options](https://flask.palletsprojects.com/en/1.1.x/deploying/).

Please be aware that **you should not have any expectation of security or performance** by using the provided development server.

### Docker

The container is based on Ubuntu 20.04. The provided scripts should be enough to build and launch the image; the code should be
pretty self-explanatory. Some things of note are:
- The image uses Python 3.7. This is mandatory due to the fact that later versions of Python are not supported by PyTorch at the
time of writing. Do not force Python 3.8+ for any reason.
- To start the container, you have three options:
  - `./start_container.sh -s`: Starts the container and the web service.
  - `./start_container.sh -d`: Starts the container and the web service, but in detached mode.
  - `./start_container.sh -i`: Starts the container and starts a bash console.
  The script will automatically mount the folder with the code. **DO NOT MOVE FILES AROUND** or the service will break.
- The other scripts are mainly for maintenance and are used to shut down the service (`remove_container.sh`) and to log into the 
container when started in detached mode (`login_running_container.sh`). 
