import cv2
import numpy as np
from scipy import stats

def cal_undistort(img, objpoints, imgpoints):
    # Image Calibration
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst, dist, mtx


def get_img_obj_points(img, nx, ny):
    imgpoints = []
    objpoints = []

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # if corners are found
    if ret == True:
        imgpoints.append(corners)
        # Draw and display the corners
        objpoints.append(objp)

    return imgpoints, objpoints


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    # 3) Take the absolute value of the derivative or gradient
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    # 6) Return this mask as your binary_output image

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


def mag_thresh(img, sobel_kernel=9, mag_thresh=(30, 100)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Calculate the magnitude
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    # 5) Create a binary mask where mag thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
    scaled_sobel = np.uint8(255 * magnitude / np.max(magnitude))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary_output


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi / 2)):
    # Apply the following steps to img
    # 1) Convert to grayscale
    # 2) Take the gradient in x and y separately
    # 3) Take the absolute value of the x and y gradients
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    # 5) Create a binary mask where direction thresholds are met
    # 6) Return this mask as your binary_output image
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    gradient_direction = np.arctan2(abs_sobely, abs_sobelx)

    binary_output = np.zeros_like(gradient_direction)
    binary_output[(gradient_direction >= thresh[0]) & (gradient_direction <= thresh[1])] = 1

    return binary_output


def sobel_filter(image, ksize=3):
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    s_channel = hls[:, :, 0]

    gradx = abs_sobel_thresh(s_channel, 'x', 10, 200)
    grady = abs_sobel_thresh(s_channel, 'y', 10, 200)

    combined = np.zeros_like(grady)
    combined_condition = ((gradx == 1) & (grady == 1))
    return combined_condition

def hls_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    l_channel = hls[:, :, 1]

    # Threshold color channel
    s_thresh_min = 120
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition




def hsv_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    s_channel = hls[:, :, 2]

    # Threshold color channel
    s_thresh_min = 160
    s_thresh_max = 255
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition




def yuv_filter(image):
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    s_channel = hls[:, :, 0]

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary_condition = (s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)
    return s_binary_condition


def rgb_filter(image):
    # Extract RG colors for better yellow line isolation
    color_threshold = 170
    R = image[:, :, 0]
    G = image[:, :, 1]
    color_combined = np.zeros_like(R)
    r_g_condition = (R > color_threshold) & (G > color_threshold)
    return r_g_condition


def filter_image(image, is_blind=False):
    sobel_condition = sobel_filter(image)
    hls_condition = hls_filter(image)
    rgb_condition = rgb_filter(image)
    hsv_condition = hsv_filter(image)
    yuv_condition = yuv_filter(image)

    height, width = image.shape[0], image.shape[1]
    # apply the region of interest mask
    combined_binary = np.zeros((height, width), dtype=np.uint8)
    if not is_blind:
        combined_binary[((rgb_condition | hsv_condition | yuv_condition) & (hls_condition | sobel_condition))] = 1
    else :
        combined_binary[sobel_condition] = 1

    mask = np.zeros_like(combined_binary)
    region_of_intersect = np.array([[0, height], [width / 2, int(0.5 * height)], [width, height]], dtype=np.int32)
    cv2.fillPoly(mask, [region_of_intersect], 1)
    thresholded = cv2.bitwise_and(combined_binary, mask)
    return thresholded


def get_curvature_radius(fit, ploty):
    x = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]
    y_eval = np.max(ploty)
    curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Obtain converted polynomials
    fit_cr = np.polyfit(ploty * ym_per_pix, x * xm_per_pix, 2)
    # Calculate converted curvature radius
    curverad = ((1 + (2 * fit_cr[0] * y_eval * ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    return curverad


def get_offset_from_center(left_x, right_x, height=720, width=1280):

    lane_center = (right_x[height-1] + left_x[height-1]) / 2
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    img_center_offset = abs(width / 2 - lane_center)
    offset_metters = xm_per_pix * img_center_offset
    return offset_metters


def get_source_points():
    return [[205,720], [1100, 720], [690, 450], [590, 450]]


def get_destination_points(width, height, fac=0.3):
    fac = 0.3
    p1 = [fac * width, height]
    p2 = [width - fac * width, height]
    p3 = [width - fac * width, 0]
    p4 = [fac * width, 0]
    destination_points = [p1,p2,p3,p4]
    return destination_points

def perspective_transform(image):
    height, width = image.shape[0], image.shape[1]
    img_size = (width,height)
    source_points = get_source_points()
    destination_points = get_destination_points(width, height)
    src = np.float32(source_points)
    dst = np.float32(destination_points)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def perspective_transform_with_filled_area(original_image, filtered_image):
    warped = perspective_transform(filtered_image)
    source_points = np.array(get_source_points())
    filled = cv2.polylines(original_image.copy(), [source_points], True, (0, 255, 0), thickness=2)
    return warped, filled


def get_lane_rectangles(warped, left_fit=None, right_fit=None, is_blind = False):
    histogram = np.sum(warped[warped.shape[0] // 2:, :], axis=0)
    histogram[600:750] = 0
    stat_left = None
    stat_right = None
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warped, warped, warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    t=0
    # Choose the number of sliding windows
    nwindows = 20
    # Set height of windows
    window_height = np.int(warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin

    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    margin_left = 100
    margin_right = 100
    min_margin = 40
    max_margin = 100

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window + 1) * window_height
        win_y_high = warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin_left
        win_xleft_high = leftx_current + margin_left
        win_xright_low = rightx_current - margin_right
        win_xright_high = rightx_current + margin_right

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]



        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)


        margin_left = max(min(margin_left+margin_left*0.10,700/(len(good_left_inds)+1)),margin_left-margin_left*0.10)
        margin_right = max(min(margin_right+margin_right*0.10,700/(len(good_right_inds)+1)),margin_right-margin_right*0.10)

        margin_left = int(min(max(min_margin, margin_left), max_margin))
        margin_right = int(min(max(min_margin, margin_right), max_margin))
        # Draw the windows on the visualization image


        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      (0, 255, 0), 2)

        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      (0, 255, 0), 2)


        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    try :
        left_fit = np.polyfit(lefty, leftx, 2)
    except Exception as ex:
        pass

    try :
        right_fit = np.polyfit(righty, rightx, 2)
    except Exception as ex:
        pass

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    return ploty, left_fitx, right_fitx, left_fit, right_fit, out_img


def get_next_frame_lines(warped, left_fit, right_fit, is_blind=False):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    margin = 20
    blind_threshold = 3800
    retrive_from_blind_threshold = 30000
    is_left_blind = is_blind
    is_right_blind = is_blind

    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                           2] + margin)))

    num_left_indices = len(left_lane_inds)
    num_right_indices = len(right_lane_inds)

    if ((num_left_indices>blind_threshold and not is_blind) or (is_blind and num_left_indices>retrive_from_blind_threshold)):
        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        is_left_blind = False
        try:
            left_fit = np.polyfit(lefty, leftx, 2)
        except Exception as ex:
            is_left_blind = True
    else:
        is_left_blind = True

    if ((num_right_indices>blind_threshold and not is_blind) or (is_blind and num_right_indices>retrive_from_blind_threshold)):
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        is_right_blind = False
        try:
            right_fit = np.polyfit(righty, rightx, 2)
        except Exception as ex:
            is_right_blind = True
    else :
        is_right_blind = True

    is_blind_current = is_right_blind or is_left_blind
    is_blind = is_blind_current
    # Generate x and y values for plotting
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return result, ploty, left_fitx, right_fitx, left_fit, right_fit, is_blind


def inverse_perspective_transform(original_image, warped, left_fitx, right_fitx, ploty):
    height, width = original_image.shape[0], original_image.shape[1]


    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    src = np.float32(get_source_points())
    dst = np.float32(get_destination_points(width, height))
    M_inverse = cv2.getPerspectiveTransform(dst, src)
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inverse, (original_image.shape[1], original_image.shape[0]))
    # Combine the result with the original image
  
    return newwarp


def add_diagnostic_image(base_image, debug_image, position):
    width_offset = base_image.shape[1] // 3
    height_offset = base_image.shape[0] // 3
    y_offset = (position // 3) * height_offset
    x_offset = (position % 3) * (width_offset)

    res = cv2.resize(debug_image, None, fx=1 / 3.25, fy=1 / 3.25, interpolation=cv2.INTER_CUBIC)

    if len(res.shape) == 2:
        base_image[y_offset:y_offset + res.shape[0], x_offset:x_offset + res.shape[1], 0] = res * 255
        base_image[y_offset:y_offset + res.shape[0], x_offset:x_offset + res.shape[1], 1] = res * 255
        base_image[y_offset:y_offset + res.shape[0], x_offset:x_offset + res.shape[1], 2] = res * 255
    else:
        base_image[y_offset:y_offset + res.shape[0], x_offset:x_offset + res.shape[1]] = res

    return base_image


def add_diagnostic_text(image, text, position, offset=150):
    width_offset = image.shape[1] // 3
    height_offset = image.shape[0] // 3
    y_position = (position // 3) * height_offset + offset
    x_position = (position % 3) * (width_offset)

    cv2.putText(image, text, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), thickness=2)

    return image
