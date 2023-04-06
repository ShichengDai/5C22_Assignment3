# 5C22_Assignment3


## High-level Description of the project
This assignment is based on the previous matlab codes aiming to use Bayesian Matting approach on a series of images and to evaluate the performance on different datasets.
The final performance is evaluated by MSE and PSNR and also compared with the Laplacian matting.

---

## Installation and Execution

ENVIRONMENT:
```sh
matplotlib==3.6.2
numpy==1.23.5
scipy==1.9.3
opencv-python==4.7.0.72
imageio==2.9.0
```

Afer installing all required packages you can run the demo file simply by typing:
```sh
python main.py
```
In order to show the performance on different images,you can use the file by typing:
 ```sh
python testing.py
```
---

## Code Features
The whole code is aimed to use Bayesian Matting on a series of images. 

The backbone of the program is `Bayesian.py`, in which the raw images and trimap are read using `imageio`. After this, forefround masks and background masks are made according to the trimap. With the help of the masks, two key variables F and B can be acquired easily.

The following part is some pre-process on the unkown regions in the trimap. Our method is to deal with the unkown pixels layer by layer. In order to deal with this issue, we introduce the `cv2.erode` fuction to help. 

For each unkown pixel, we draw a block to extract the features surrounding the particular picxel. This function is impled inside `block.py`. And we consider that if there are not adequate pixel in the purpose block, we will add some paddings to solve this.

After drawing a block following a customized variable N(block size), we also use 2D gaussian filter to calculate the weight for each pixel inside the block. This function is made in `gaussian_filter.py`.

Also inside the loop of finding alpha values, we set a threshold for the number of background values and foreground values inside a block to make each step meaningful. However, with this limitation, the loop will be stucked in some occassions. Thus we add some extra lines to allow the N(block size) to increase and the threshold to decrease at the meanwhile.

All the previous introductions are some pre-process for the given image. Here are three core functions to solve the problem.

**Clustering Function**

There are two main functions in clustering part.

The clustFunc function takes a measurements vector S and a weights vector w as input, and initializes a list of nodes with a single Node object containing the entire matrix S and weights w. The function then repeatedly splits the node with the largest eigenvalue (lmbda) until the maximum eigenvalue falls below a threshold (minVar). After all nodes have been split, the function returns the mean (mu) and covariance (cov) of each node as arrays.

The split function takes a list of Node objects as input, identifies the node with the largest eigenvalue, splits the node into two nodes using the eigenvector (e) associated with the largest eigenvalue, and returns the updated list of nodes.


**Solve Function**  
The `solve.py` implements a function `solve` that calculates the F, B and alpha values based on the bayesian matting paper quadratic equations:
It takes the wighted mean `mu_F` , `mu_B` , `Sigma_B` and `sigma_C` as inputs and calculate the most likely estimates of F, B and alpha with a maximum a posteriori technique.
we break the problem into two quadratic equations.  
1.  In the first euation we assume the alpha is constant.  
<img src="Bayes_Output\alpha_equation.PNG" width="300" >  
2. In second equation we consider the F and B are constant.  
<img src="Bayes_Output\f_and_b_equation.PNG" width="300" >  

The likelihood of the given foreground, background, and alpha matte is calculated at for each iteration, until the maximum number of iterations is reached or diffrence is less than the `minLike`.

The function returns the foreground `max_F`, the background `max_B`, and the alpha values `max_alpha`. The alpha values are the transparency of the foreground, with a value of 1 represents the background image and 0 indicates foreground.

**Unittest Functions**

The class named `bayesiantesting`, has three unit test cases for a Bayesian testing function that works with image data. The three test cases are as follows:

1. `test_checkshape()`: This test case checks if the shape of the ground truth (GT) image matches the shape of the predicted alpha matte i.e. height and width of the alpha matte.
2. `test_alphavalues()`: This test case checks if the alpha values are within the range 0 to 1 predicted by the function are correct.
3. `test_check_Comp_shape()`: This test case checks if the shape of the ground truth (GT) image matches the shape of the predicted composite image i.e. height and width of the alpha matte.





**Results**

Here are all evaluation for every output. The input raw images here have complex background. In other to see if this program is robust for constant background situations, several other experiments are made.
|      | Time_bayesian (sec) | Laplacian_MSE | Laplacian_PSNR | Bayesian_MSE | Bayesian_PSNR |
|------|---------------------|---------------|----------------|--------------|---------------|
| GT01 | 244.33              | 0.039039882   | 14.08491502    | 0.000797091  | 30.98492176   |
| GT02 | 529.62              | 0.035916205   | 14.44709559    | 0.003942424  | 24.04236689   |
| GT03 | 4947.37             | 0.144069803   | 8.414270383    | 0.004339649  | 23.62545408   |
| GT04 | 2889.02             | 0.141887671   | 8.4805534      | 0.004961305  | 23.04404075   |
| GT05 | 220.33              | 0.028728009   | 15.41694474    | 0.001418964  | 28.48028623   |
| GT06 | 459.45              | 0.04903508    | 13.0949311     | 0.00222974   | 26.51745845   |
| GT07 | 449.25              | 0.042396067   | 13.72674433    | 0.000665663  | 31.76745577   |
| GT08 | 817.06              | 0.115818595   | 9.36221706     | 0.008272551  | 20.82360566   |
| GT09 | 721.61              | 0.08256032    | 10.8322863     | 0.003595138  | 24.44284466   |
| GT10 | 501.41              | 0.04714589    | 13.26556165    | 0.001702608  | 27.68885283   |
| GT11 | 320.03              | 0.054783306   | 12.61351763    | 0.002155691  | 26.66413501   |
| GT12 | 172.56              | 0.034092204   | 14.67344924    | 0.000587343  | 32.31107911   |


As shown here are some output alpha maps. The fisrt pair is the comparison between the output for GT04 and its ground truth. It seems that if there is a solid background, then Bayesian matting will not act properly in some complex areas(hairs or considering the transparence).

<img src="Bayes_Output\GT04.png" width="350" >  <img src="Image_Source\Ground_Truth\GT04.png" width="350" >

The second pair is the comparison between the output for GT04 with constant background and its ground truth.

<img src="Bayes_Output\const_GT04.png" width="350" >  <img src="Image_Source\Ground_Truth\GT04.png" width="350" >

The visual quality of the second one is worse than the first one and also can be seen in the table below which shows the results from constant background images. However, the performance is worse the the former experiment, the running speed is higher than the first one. And as seen in the following two images are the composited imges from the two experiments(Left: constant background; Right: complex background).

|      | Time_bayesian (sec) | Laplacian_MSE | Laplacian_PSNR | Bayesian_MSE | Bayesian_PSNR |
|------|---------------------|---------------|----------------|--------------|---------------|
| GT01 | 173.18              | 0.039223591   | 14.06452651    | 0.00239683   | 26.20362705   |
| GT02 | 266.69              | 0.035110048   | 14.54568574    | 0.007790774  | 21.08419392   |
| GT03 | 4244.61             | 0.142706174   | 8.455572377    | 0.049884994  | 13.02030073   |
| GT04 | 2112.79             | 0.140667385   | 8.518065866    | 0.040326065  | 13.94414151   |

<img src="Basic_matting\composite_output\comp_GT04_const.png" width="350" >  <img src="Basic_matting\composite_output\comp_GT04.png" width="350" >

---
## Credits
Rubeinstein - Matlab  
Marco forte - Python  
This code was developed for purely academic purposes by Shicheng Dai， Dian Zhuang and Atul Redekar as part of the module 5C22 Compuntational Methods. we have taken a You can get access to the codes via https://github.com/ShichengDai/5C22_Assignment3

