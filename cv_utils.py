from skimage.feature import hog
import cv2
import numpy as np
import matplotlib.image as mpimg

# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                                  transform_sqrt=True,
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm= 'L2-Hys',
                       transform_sqrt=True,
                       visualize=vis, feature_vector=feature_vec)
        return features

def convert_color_space(img, color_space='RGB'):
    # Convert image to new color space (if specified)
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    return feature_image
    # Use cv2.resize().ravel() to create the feature vector


def bin_spatial(img, size=(32, 32)):
    # Convert image to new color space (if specified)
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hist_range=(0,255)):
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    feature_image = convert_color_space(image, color_space=color_space)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)


def get_image_features(images, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True, hist_range=(0,255)):

    all_features = []
    for image in images:
        image =  mpimg.imread(image)
        features = single_img_features(image, color_space=color_space, spatial_size=spatial_size,
                            hist_bins=hist_bins, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat, hist_range=hist_range)
        all_features.append(features)


    return all_features


def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, 450:, :]
    ctrans_tosearch = convert_color_space(img_tosearch, color_space='YUV')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = \
            cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    rectangles = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            #subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            #             spatial_features = bin_spatial(subimg, size=spatial_size)
            #             hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                rectangles.append(
                    ((xbox_left+450, ytop_draw + ystart), (xbox_left + win_draw+450, ytop_draw + win_draw + ystart)))

    return rectangles


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap



class Rectangle:
    """
    Create robust rectangle update based on creating animation approximating to another bbox that least for few frames   
    
    Class members:
    updating -- Whether the animation for approximating to a bbox still performing or not   
    tick_count -- Frames till the animation will still perform
    remove_count -- Frames till the rectangle will still be alive
    start_count -- Frames till the rectangle will start be visible live
    """

    updating = False
    tick_count = 0
    remove_count = 30
    start_count = 5

    def __init__(self, x1, y1, x2, y2, _id):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.id = _id
        self.updating = False
        self.start_count = 5

    def update(self, x1, y1, x2, y2, frames=0):
        remove_count = 20
        self.tick_count = frames

        if self.start_count != 0:
            self.start_count -= 1
        else:
            self.updating = True

        distX1 = x1 - self.x1
        distX2 = x2 - self.x2
        distY1 = y1 - self.y1
        distY2 = y2 - self.y2

        self.stepX1 = distX1 / frames
        self.stepX2 = distX2 / frames
        self.stepY1 = distY1 / frames
        self.stepY2 = distY2 / frames

        return True

    def update_delta(self):

        if self.updating:
            self.x1 += self.stepX1
            self.x2 += self.stepX2
            self.y1 += self.stepY1
            self.y2 += self.stepY2

            self.tick_count -= 1

        #the animation is finished
        if self.tick_count == 0:
            self.updating = False

    def get_distance(self, x1, y1, x2, y2):
        return abs(self.x1 - x1) + abs(self.y1 - y1) + abs(self.x2 - x2) + abs(self.y2 - y2)


def draw_labeled_bboxes(img, labels, car_dict):

    # Iterate through all detected cars
    car_set = set()
    for iterator_number in range(1, labels[1] + 1):
        car_number = iterator_number
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        # print(car_dict)

        x1 = np.min(nonzerox)
        y1 = np.min(nonzeroy)
        x2 = np.max(nonzerox)
        y2 = np.max(nonzeroy)

        min_distance = 2**31 - 1
        min_key = None

        #get min rectangle distance
        for key in car_dict:
            cur_dist = car_dict[key].get_distance(x1, y1, x2, y2)

            if cur_dist < min_distance:
                min_distance = cur_dist
                min_key = key

        # if we found similar rectangle from previous frame
        if min_key is not None:
        	# if minimum distance is less than 300, then we found rectangle from previous frame
            if min_distance < 300:
                car_number = min_key
            else:
            	#fill where we have empty place (todo code improvement : we can remove the for iteration)
                for i in range(1, 10):
                    if i not in car_dict:
                        car_number = i
                        break

        # print('car_number', car_number, 'min_dist', min_distance, 'minkey', min_key)
        car_set.add(car_number)

        if car_number in car_dict:
            rect = car_dict[car_number]
            if rect.updating:
                # the bbox is updating, and we will proceed to the next update frame part
                rect.update_delta()
                bbox = ((int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)))

            else:
                # the bbox is not updating, and we should start approximating 
                bbox = ((int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)))
                rect.update(x1, y1, x2, y2, frames=20)
        else:
        	# We create initial bounding box
            bbox = ((x1, y1), (x2, y2))
            if (abs(x1 - x2) >= abs(y1 - y2)):
                car_dict[car_number] = Rectangle(x1, y1, x2, y2, car_number)

        # Draw the box on the image
        if car_number in car_dict:
        	if car_dict[car_number].start_count == 0:
        		cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
        else:
            # print('not drawn', car_number, 'count:', car_dict[car_number].start_count)
            pass
        # print('car_nymber', car_number, 'tick_count', rect.tick_count,'updating', rect.updating,'remove count', rect.remove_count,
        # 'start_count', rect.start_count)
    # Return the image
    car_keys = car_dict.keys()

    #find not used rectangles in this frame
    exclusion = car_keys - car_set
    for key in exclusion:
        rect = car_dict[key]
        #increase the remove count
        rect.remove_count -= 1
        if not rect.updating:
        	#reset the start count
            rect.start_count = 5
        if (rect.remove_count == 0):
            del car_dict[key]
            continue
        if rect.updating:
        	#if the rectangle is in updating state draw the cv rectangle
            bbox = ((int(rect.x1), int(rect.y1)), (int(rect.x2), int(rect.y2)))
            cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)

    return img, car_dict