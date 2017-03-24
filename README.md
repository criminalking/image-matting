# Image Matting

This is a c++ implementation of the paper *Image Matting with Local and Nonlocal Smooth Priors* by Xiaowu Chen et al.

## What is Image Matting?

Mathematically, the image *I* is a linear combination of *F* and *B* as the following: *I=Fα+B(1-α)*. Here, the alpha matte α defines the opacity of each pixel and its value lies in [0,1]. Our goal is to estimate this α. In this paper, sampling and affinity matting are combined to solve this problem. Sampling-Based Matting estimates the alpha matte, foreground and background color of a pixel simultaneously. Affinity-Based Matting solves the alpha matte independent of the foreground and background colors which consists of local smooth prior and nonlocal smooth prior.

## Pre-requisite

* [opencv 2.4.9](http://opencv.org/)
* [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page)

## Usage

* build: make build
* run: ./main <input_image_in_pic> <input_trimap_in_pic> <output_image_in_results>

## Configuration

* System: Mac OS X
* Programming language: c++

## Results

![original image](/pic/lowpic.png)
![trimap image](/pic/lowtrimap.png)
![result image](/results/result_low.png)

## Future Work

* Actually, many results have artifacts(worse than this paper's results), such as ![artifacts](/results/result_plant.jpg). Should think about the parameters or change a new implementation in some details.

## Thanks

* Date: 5-2016
* Author: criminalking
