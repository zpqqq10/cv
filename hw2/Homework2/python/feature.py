import os
import cv2
import numpy as np
from tqdm import trange

class Processor:
    def __init__(self, data):
        self.colors = data['color']
        self.grays = data['gray']
        self.sift = cv2.SIFT_create()
        self.keypoints = {}
        self.descriptors = {}
        self.matches = {}
        
    def detect(self):
        for dataset, data in self.grays.items():
            self.keypoints[dataset] = {}
            for image_name, image in data.items():
                # tuple of n features
                self.keypoints[dataset][image_name] = self.sift.detect(image, None)
        print('feature detection done')
    
    def describe(self):
        for dataset, data in self.grays.items():
            self.descriptors[dataset] = {}
            for image_name, image in data.items():
                # array of n*128 descriptors, each feature described by 128-d vector
                _, self.descriptors[dataset][image_name] = self.sift.compute(image, self.keypoints[dataset][image_name])
        print('feature description done')
                
    def match(self, des1, des2, threshold = .7):
        def l2_matcher(des1, des2, threshold = .7):
            matches = []
            for i in trange(des1.shape[0]):
                min_dist = 1e10
                second_min_dist = 1e10
                min_idx = 0
                for j in range(des2.shape[0]):
                    # distance between des1[i] and des2[j]
                    dist = np.sqrt(np.sum((des1[i] - des2[j])**2))
                    if dist < min_dist:
                        second_min_dist = min_dist
                        min_dist = dist
                        min_idx = j
                    elif dist < second_min_dist:
                        second_min_dist = dist
                
                # ratio test
                if min_dist/second_min_dist < threshold:
                    matches.append(cv2.DMatch(i, min_idx, min_dist))
            return matches
        
        return l2_matcher(des1, des2, threshold)
    
    def compute_homography(self, dataset, img1, img2):
        # compute matches
        print(f'matching {dataset}/{img1} and {dataset}/{img2}')
        if os.path.exists(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts1.npy')) and os.path.exists(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts2.npy')):
            pts1 = np.load(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts1.npy'))
            pts2 = np.load(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts2.npy'))
        elif os.path.exists(os.path.join('..', 'result', f'{dataset}_{img2}_{img1}_pts1.npy')) and os.path.exists(os.path.join('..', 'result', f'{dataset}_{img2}_{img1}_pts2.npy')):
            pts1 = np.load(os.path.join('..', 'result', f'{dataset}_{img2}_{img1}_pts2.npy'))
            pts2 = np.load(os.path.join('..', 'result', f'{dataset}_{img2}_{img1}_pts1.npy'))
        else: 
            matches = self.match(self.descriptors[dataset][img1], self.descriptors[dataset][img2])

            # extract points
            pts1 = np.float32([self.keypoints[dataset][img1][match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([self.keypoints[dataset][img2][match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
            np.save(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts1.npy'), pts1)
            np.save(os.path.join('..', 'result', f'{dataset}_{img1}_{img2}_pts2.npy'), pts2)
        
        print(f'computing homography between {dataset}/{img1} and {dataset}/{img2}')
        homography_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=3000)

        return homography_matrix
    
    def panorama_stitch(self, _path):
        self.detect()
        
        self.describe()
        
        for dataset in self.colors.keys():
            # compute homography between 
            img_list = list(self.colors[dataset].keys())
            img_list.sort()
            # 选择中间的图像作为标准
            standard_idx = 1
            standard_img = img_list[standard_idx]
            
            panorama = self.colors[dataset][standard_img]
            
            if dataset == 'data1': 
                standard_idx = 3
                standard_img = img_list[standard_idx]
                
                panorama = self.colors[dataset][standard_img]
                # # 右侧图像拼接
                img2add = self.colors[dataset][img_list[2]]
                homography = self.compute_homography(dataset, img_list[2], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                # # 下侧图像拼接
                img2add = self.colors[dataset][img_list[0]]
                homography = self.compute_homography(dataset, img_list[0], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'down')
                img2add = self.colors[dataset][img_list[1]]
                homography = self.compute_homography(dataset, img_list[1], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'down', expand=False)
                # 上侧图像拼接
                img2add = self.colors[dataset][img_list[4]]
                homography = self.compute_homography(dataset, img_list[4], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'up', displace=[1, 0])
                img2add = self.colors[dataset][img_list[5]]
                homography = self.compute_homography(dataset, img_list[5], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'up', displace=[2, 0])
                cv2.imwrite(os.path.join(_path, f'{dataset}_panorama.jpg'), panorama[img2add.shape[0]:, :])
                
            elif dataset == 'data2': 
                standard_idx = 2
                standard_img = img_list[standard_idx]
                
                panorama = self.colors[dataset][standard_img]
                # 右侧图像拼接
                img2add = self.colors[dataset][img_list[1]]
                homography = self.compute_homography(dataset, img_list[1], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                img2add = self.colors[dataset][img_list[0]]
                homography = self.compute_homography(dataset, img_list[0], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                # 左侧图像拼接
                img2add = self.colors[dataset][img_list[3]]
                homography = self.compute_homography(dataset, img_list[3], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'left', displace=[0, 1])
                cv2.imwrite(os.path.join(_path, f'{dataset}_panorama.jpg'), panorama)
                
            elif dataset == 'data3': 
                # 右侧图像拼接
                img2add = self.colors[dataset][img_list[2]]
                homography = self.compute_homography(dataset, img_list[2], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                # 左侧图像拼接
                img2add = self.colors[dataset][img_list[0]]
                homography = self.compute_homography(dataset, img_list[0], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'left', displace=[0, 1])
                cv2.imwrite(os.path.join(_path, f'{dataset}_panorama.jpg'), panorama)
            else: 
                # 右侧图像拼接
                img2add = self.colors[dataset][img_list[2]]
                homography = self.compute_homography(dataset, img_list[2], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                img2add = self.colors[dataset][img_list[3]]
                homography = self.compute_homography(dataset, img_list[3], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'right')
                # 左侧图像拼接
                img2add = self.colors[dataset][img_list[0]]
                homography = self.compute_homography(dataset, img_list[0], standard_img)
                panorama = self.stitch_images(panorama, img2add, homography, 'left', displace=[0, 1])
                cv2.imwrite(os.path.join(_path, f'{dataset}_panorama.jpg'), panorama)
                
            
    def stitch_images(self, img1, img2, homography, direction, displace = [0, 0], expand = True):
        '''stitch a new image 

        Args:
            img1 (_type_): _description_
            img2 (_type_): _description_
            homography (_type_): _description_
            direction (str): in ['up', 'down', 'left', 'right']
            displace (array): how many offset (horizontal: img2.shape[1] * displace[1], vertical: img2.shape[0] * displace[0]) is made
            expand (bool): whether to expand img1

        Returns:
            warpImg: the img2 being warped and stitched to img1 
        '''
        offset = 0
        # Warp the second image onto the first image
        homography = np.dot(np.array([[1, 0, img2.shape[1]*displace[1]], [0, 1, img2.shape[0]*displace[0]], [0, 0, 1]]), homography)
        if direction == 'left':
            offset = img2.shape[1]
            if expand: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            else: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))
        elif direction == 'right': 
            # offset = img2.shape[1]
            # homography = np.dot(np.array([[1, 0, offset], [0, 1, 0], [0, 0, 1]]), np.linalg.inv(homography))
            offset = 0
            if expand: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1] + img2.shape[1], img1.shape[0]))
            else: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))
        elif direction == 'up':
            offset = img2.shape[0]
            if expand: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0] + img2.shape[0]))
            else: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))
        elif direction == 'down':
            if expand: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0] + img2.shape[0]))
            else: 
                warpImg = cv2.warpPerspective(img2, homography, (img1.shape[1], img1.shape[0]))
        else: 
            raise NotImplementedError

        # Find the overlapping region
        rows, cols = img1.shape[:2]
        left = 0
        right = 0
        top = 0
        bottom = 0
        if direction in ['left', 'right']:
            for col in range(0, cols):
                if img1[:, col].any() and warpImg[:, col+offset].any():
                    left = col
                    break
            for col in range(cols - 1, 0, -1):
                if img1[:, col].any() and warpImg[:, col+offset].any():
                    right = col
                    break
        else : 
            for row in range(0, rows):
                if img1[row, :].any() and warpImg[row+offset, :].any():
                    top = row
                    break
            for row in range(rows - 1, 0, -1):
                if img1[row, :].any() and warpImg[row+offset, :].any():
                    bottom = row
                    break

        # Create a new image with the size of the two images combined
        res = np.zeros([rows, cols, 3], np.uint8)
        if direction in ['left', 'right']:
            for row in range(0, rows):
                for col in range(0, cols):
                    if not img1[row, col].any():#如果没有原图，用旋转的填充
                        res[row, col] = warpImg[row, col+offset]
                    elif not warpImg[row, col+offset].any():
                        res[row, col] = img1[row, col]
                    else:
                        srcImgLen = float(abs(col - left))
                        testImgLen = float(abs(col - right))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(img1[row, col] * (1-alpha) + warpImg[row, col+offset] * alpha, 0, 255)
        else: 
            for row in range(0, rows):
                for col in range(0, cols):
                    if not img1[row, col].any():
                        res[row, col] = warpImg[row+offset, col]
                    elif not warpImg[row+offset, col].any():
                        res[row, col] = img1[row, col]
                    else:
                        srcImgLen = float(abs(row - top))
                        testImgLen = float(abs(row - bottom))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(img1[row, col] * (1-alpha) + warpImg[row+offset, col] * alpha, 0, 255)

       
        if direction in ['left', 'right']:
            warpImg[0:img1.shape[0], offset:offset+img1.shape[1]] = res
        else: 
            warpImg[offset:offset+img1.shape[0], 0:img1.shape[1]] = res

        return warpImg
            
    def visualize_matches(self, _path):
        matches = self.match(self.descriptors['data1']['112_1298.JPG'], self.descriptors['data1']['112_1299.JPG'])
        img_matches = cv2.drawMatches(self.colors['data1']['112_1298.JPG'], self.keypoints['data1']['112_1298.JPG'],
                                      self.colors['data1']['112_1299.JPG'], self.keypoints['data1']['112_1299.JPG'],
                                      matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imwrite(os.path.join(_path, 'match_data1.jpg'), img_matches)
    
    def visualize_keypoints(self, _path):
        for dataset, data in self.colors.items():
            for image_name, image in data.items():
                img = cv2.drawKeypoints(image, self.keypoints[dataset][image_name], None)
                cv2.imwrite(os.path.join(_path, f'{image_name}.jpg'), img)
        return 
    
    def visualize_comparison(self, _path):
        self.detect()
        self.describe()
        MAXITERS = 100
        datas = [['data2', 'IMG_0488.JPG', 'IMG_0489.JPG'], 
                 ['data3', 'IMG_0675.JPG', 'IMG_0676.JPG'],
                 ['data4', 'IMG_7356.JPG', 'IMG_7357.JPG']]
        # data2
        # Inlier Ratio of SIFT: 0.7040704070407041 (640/909)
        # Inlier Ratio of pixel values: 0.072 (9/125)
        # data3
        # Inlier Ratio of SIFT: 0.7839643652561247 (704/898)
        # Inlier Ratio of pixel values: 0.050314465408805034 (8/159)
        # data4
        # Inlier Ratio of SIFT: 0.5617597292724196 (1660/2955)
        # Inlier Ratio of pixel values: 0.014787430683918669 (8/541)
        for dataset, img1, img2 in datas:
            print(f'matching {dataset}/{img1} and {dataset}/{img2}')
            matches = self.match(self.descriptors[dataset][img1], self.descriptors[dataset][img2])

            # extract points
            pts1 = np.float32([self.keypoints[dataset][img1][match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([self.keypoints[dataset][img2][match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
            
            print(f'computing homography between {dataset}/{img1} and {dataset}/{img2}')
            homography_matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=MAXITERS)
            inlier_ratio = np.sum(mask) / len(mask)
            print(f"Inlier Ratio of SIFT: {inlier_ratio} ({np.sum(mask)}/{len(mask)})")
            img_matches = cv2.drawMatches(self.colors[dataset][img1], self.keypoints[dataset][img1], 
                                        self.colors[dataset][img2], self.keypoints[dataset][img2], matches, None, matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(_path, f'sift_match_{dataset}.jpg'), img_matches)
            
            self.descriptors[dataset] = {}
            image = self.grays[dataset][img1]
            self.descriptors[dataset][img1] = []
            for kp in self.keypoints[dataset][img1]:
                # concatenate 3*3 patch
                x, y = int(kp.pt[0]), int(kp.pt[1])
                self.descriptors[dataset][img1].append(image[max(0, y-1):min(image.shape[0], y+2), max(0, x-1):min(image.shape[1], x+2)].flatten())
            self.descriptors[dataset][img1] = np.array(self.descriptors[dataset][img1])
            image = self.grays[dataset][img2]
            self.descriptors[dataset][img2] = []
            for kp in self.keypoints[dataset][img2]:
                # concatenate 3*3 patch
                x, y = int(kp.pt[0]), int(kp.pt[1])
                self.descriptors[dataset][img2].append(image[max(0, y-1):min(image.shape[0], y+2), max(0, x-1):min(image.shape[1], x+2)].flatten())
            self.descriptors[dataset][img2] = np.array(self.descriptors[dataset][img2])
            
            # l2 match
            matches = self.match(self.descriptors[dataset][img1], self.descriptors[dataset][img2])
            pts1 = np.float32([self.keypoints[dataset][img1][match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([self.keypoints[dataset][img2][match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
            print(f'computing homography between {dataset}/{img1} and {dataset}/{img2}')
            homography_matrix, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=MAXITERS)
            inlier_ratio = np.sum(mask) / len(mask)
            print(f"Inlier Ratio of pixel values: {inlier_ratio} ({np.sum(mask)}/{len(mask)})")
            img_matches = cv2.drawMatches(self.colors[dataset][img1], self.keypoints[dataset][img1], 
                                        self.colors[dataset][img2], self.keypoints[dataset][img2], matches, None, matchesMask=mask.ravel().tolist(), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imwrite(os.path.join(_path, f'concatenated_match_{dataset}.jpg'), img_matches)