# gan4cats
Cat faces generation with adversarial techniques.


#### Install nvidia-docker on EC2

Select AMI:
**Deep Learning AMI CUDA 9 Ubuntu Version - ami-b8c807c0**

    wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker_1.0.1-1_amd64.deb
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
    sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
    sudo apt-get update
    sudo apt-get -f install
    sudo apt-get install -y docker-ce
    sudo dpkg -i nvidia-docker*.deb
