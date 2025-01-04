# ChirpTrack - Classification of bird species through their unique vocal patterns

The goal is to build a multi-class classification model that takes a short audio file as input and outputs the predicted bird species. The project will also include the development of a simple web-based interface where users can upload an audio file, and the model will return the classification result.

In the future this project could be extended to a real-time acoustic monitoring system to detect the presence of species. By using machine learning, the system will automatically classify bird sounds in recordings obtained from remote microphones in habitats such as forests in order to track biodiversity and species health.

## Team members

- David Hodel (<david.hodel@stud.hslu.ch>)
- Maiko Trede (<maiko.trede@stud.hslu.ch>)
- Nevin Helfenstein (<nevin.helfenstein@stud.hslu.ch>)

## Getting Started

1. Make sure you have [Git LFS](https://git-lfs.github.com/) installed
2. Make sure you have installed [Miniconda](https://docs.anaconda.com/miniconda/) or [Anaconda](https://www.anaconda.com/products/distribution)
3. Create a new conda environment using the saved environment: `conda env create -f environment.yml`
4. Activate the environment: `conda activate dspro1`

## GPU Hub

To open and run the project in HSLU's GPU Hub, follow these steps:

1. Ensure that you are in the HSLU WLAN or VPN
2. Open GPUHub: <https://gpuhub.labservices.ch/>
3. Launch Server (`Minimal environment` suffices for non-DL tasks, else `PyTorch` is required)
4. Clone the repository:
  1. Open a terminal in GPUHub
  2. Create a ssh key: `ssh-keygen -t ed25519`
  3. Print the public ssh key in the terminal and manually copy the entire line: `cat ~/.ssh/id_ed25519.pub`
  4. Open GitHub, select the user icon in the upper right corner. In _Settings_ / _SSH and GPC keys_, create new ssh key and add the copied key from GPUHub to the textbox.
  6. Back in GPUHub, clone the repository: `git clone git@github.com:Davee02/HSLU.DSPRO1.BirdDetection.git`
