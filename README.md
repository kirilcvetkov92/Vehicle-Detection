## Writeup Template
### Writup for this project

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./YoloV2/notebook_images/architecture.png
[image2]: ./YoloV2/notebook_images/clustering.png
[image3]: ./YoloV2/notebook_images/iou.png
[image4]: ./YoloV2/notebook_images/nms_algo.jpg
[image5]: ./YoloV2/notebook_images/probability_extraction.png
[image6]: ./YoloV2/notebook_images/rectangle.png
[image7]: ./YoloV2/notebook_images/yolo.png
[image8]: ./YoloV3/notebook_images/architecture.png
[image9]: ./YoloV3/notebook_images/formula.png
[image10]: ./documentation/hog.png


### Histogram of Oriented Gradients (HOG)

#### 1. Explanation how I extracted HOG features from the training images.


I started by reading in all the `vehicle` and `non-vehicle` images. 

First and foremost we need to extract several types of features from the image and concatenate them into one vector, representing the image in the multi-dimensional space, ready for classification

The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.

The hog features extractor is supported in `skimage` framework and it is implemented in `cv_utils.py:get_hog_features (Line:17 - Line:32)
     
Here is an example using the `YUV` color space and HOG parameters : 

![alt text][image10]

#### 2. How you I settled my final choice of HOG parameters.

I tried various combinations of parameters, some parameters were bringing good results in classification, but decreasing the train/prediction time and vice versa, so the parameters below brought me :
* validation accuracy : 98.6 
* fast train/test time

Bellow are the parameter combination I used for extracting the hog features for my project.                   
* Picture Color Space = 'YUV'
* Orient = 11
* Pixels per cell = (16,16)
* Cells per block = (2,2)

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using only the `hog features`. First and foremost my goal was to achieve 10+ fps, with fair performance on my Intel i5 8600K.

The full code is inside the main Notebook.py file, under `Generate dataset and train the model heading`.
I scalled the features values since the range of all features should be normalized so that each feature contributes approximately proportionately to the final distance.
Then I used the Linear SVC method which is in fact sklearn implementation for Linear Support Vector Classification.
The classifier is implemented so it has more flexibility in the choice of penalties and loss functions and should scale better to large numbers of samples.


Code: 
``
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
``    
    
    

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

