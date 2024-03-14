import cv2
import math
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from flask_cors import CORS
from ultralytics import YOLO
from flask import Flask, request, jsonify
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
CORS(app)
class MeibomianGlandAnalyzer:
    def __init__(self, image, eyelid_detector, gland_detector,
                 eyelid_detector_conf = 0.1,
                 gland_detector_conf = 0.2,
                 margin = 0.2,
                 min_gland_distance = 20):
      request = urllib.request.urlopen(image)
      arr = np.asarray(bytearray(request.read()), dtype=np.uint8)
      self.image = cv2.imdecode(arr, -1)  # 'Load it as it is'
      self.path = image
    #   self.image = cv2.imread(im)
      self.eyelid_detector = YOLO(eyelid_detector)
      self.gland_detector = YOLO(gland_detector)
      self.margin = margin
      self.min_gland_distance = min_gland_distance
      self.eyelid_detector_conf = eyelid_detector_conf
      self.gland_detector_conf = gland_detector_conf
      self.eyelid_detector_cls = None
      self.eyelid_detector_xywh = None
      self.eyelid_crop_idx = None
      self.eyelid_crop_image = None
      self.eyelid_bbox_image = None
      self.gland_images = None
      self.gland_list = None
      self.eyelid_count = 0
      self.theta = None
      self.abnormality_score = [0,0]

    def detect_eyelid(self):
      results = self.eyelid_detector.predict(source=self.image, conf=self.eyelid_detector_conf)
      self.eyelid_detector_cls = np.array(results[0].boxes.cls)
      self.eyelid_detector_conf = np.array(results[0].boxes.conf)
      self.eyelid_detector_xywh = np.round(np.array(results[0].boxes.xywh)).astype('int')
      eyelid_candidate_idx = np.where(self.eyelid_detector_cls == 0)[0];
      eyelid_candidate_area = np.prod(self.eyelid_detector_xywh[eyelid_candidate_idx,2:4],axis = 1) + self.eyelid_detector_conf[eyelid_candidate_idx]
      eyelid_xywh = self.eyelid_detector_xywh[np.argmax(eyelid_candidate_area),:]
      left_lim =  int(eyelid_xywh[0] - eyelid_xywh[2]/2)
      right_lim =  int(eyelid_xywh[0] + eyelid_xywh[2]/2)
      upper_lim = int(eyelid_xywh[1] - eyelid_xywh[3]/2)
      lower_lim = int(eyelid_xywh[1] + eyelid_xywh[3]/2)
      self.eyelid_crop_idx = [upper_lim, lower_lim, left_lim, right_lim]
      self.eyelid_crop_image = self.image[upper_lim:lower_lim,left_lim:right_lim,:]

      self.eyelid_bbox_image = self.image.copy()
      for i in range(len(self.eyelid_detector_cls)):
        if(self.eyelid_detector_cls[i] == 1 and
          self.eyelid_detector_xywh[i,0] > left_lim + round(self.image.shape[1]*self.margin) and
           self.eyelid_detector_xywh[i,0] < right_lim - round(self.image.shape[1]*self.margin) and
           self.eyelid_detector_xywh[i,1] > upper_lim and
           self.eyelid_detector_xywh[i,1] < lower_lim):
          left =  int(self.eyelid_detector_xywh[i,0] - self.eyelid_detector_xywh[i,2]/2)
          right =  int(self.eyelid_detector_xywh[i,0] + self.eyelid_detector_xywh[i,2]/2)
          upper = int(self.eyelid_detector_xywh[i,1] - self.eyelid_detector_xywh[i,3]/2)
          lower = int(self.eyelid_detector_xywh[i,1] + self.eyelid_detector_xywh[i,3]/2)
          cv2.rectangle(self.eyelid_bbox_image, (left, upper), (right,lower), color=(255,0,0), thickness=2)
          self.eyelid_count = self.eyelid_count+1;

    def detect_gland(self):
      self.gland_images = [np.zeros(self.image.shape), self.image.copy()]
      gland_coor_conf = []
      for i in range(len(self.eyelid_detector_cls)):
        if(self.eyelid_detector_cls[i] == 1 and
           self.eyelid_detector_xywh[i,0] > self.eyelid_crop_idx[2] + round(self.image.shape[1]*self.margin) and
           self.eyelid_detector_xywh[i,0] < self.eyelid_crop_idx[3] - round(self.image.shape[1]*self.margin) and
           self.eyelid_detector_xywh[i,1] > self.eyelid_crop_idx[0] and
           self.eyelid_detector_xywh[i,1] < self.eyelid_crop_idx[1]):

          left =  int(self.eyelid_detector_xywh[i,0] - self.eyelid_detector_xywh[i,2]/2)
          right =  int(self.eyelid_detector_xywh[i,0] + self.eyelid_detector_xywh[i,2]/2)
          upper = int(self.eyelid_detector_xywh[i,1] - self.eyelid_detector_xywh[i,3]/2)
          lower = int(self.eyelid_detector_xywh[i,1] + self.eyelid_detector_xywh[i,3]/2)
          im_eyelidsb_crop = self.image[upper:lower,left:right,:]

          results_sb = self.gland_detector.predict(source=im_eyelidsb_crop, conf=self.gland_detector_conf)
          xywh_sb = np.round(np.array(results_sb[0].boxes.xywh)).astype('int')
          conf_sb = np.array(results_sb[0].boxes.conf)

          temp = np.column_stack((left+xywh_sb[:, 0], upper+xywh_sb[:, 1], conf_sb))
          gland_coor_conf.append(temp)

          for j in range(len(xywh_sb)):
            self.gland_images[0] = cv2.circle(self.gland_images[0], (left+xywh_sb[j,0],upper+xywh_sb[j,1]), 5, (0,255,0), 2)
            self.gland_images[1] = cv2.circle(self.gland_images[1], (left+xywh_sb[j,0],upper+xywh_sb[j,1]), 9, (0,255,0), 1)

      gland_coor_conf = np.vstack(gland_coor_conf)
      sort_idx = gland_coor_conf[:, 2].argsort()
      sort_idx = sort_idx[::-1]
      gland_coor_conf = gland_coor_conf[sort_idx]

      indices_to_keep = []
      mask = np.ones(gland_coor_conf.shape[0], dtype=bool)
      filter_data = gland_coor_conf
      for i in range(gland_coor_conf.shape[0]):
          distance = ((gland_coor_conf[i, 0] - gland_coor_conf[:, 0])**2 + (gland_coor_conf[i, 1] - gland_coor_conf[:, 1])**2)**0.5
          close_points = distance <= self.min_gland_distance
          if np.any(close_points):
              close_points_c_values = gland_coor_conf[close_points, 2]
              if gland_coor_conf[i, 2] < np.max(close_points_c_values):
                  mask[i] = False

      indices_to_keep = np.where(mask)[0]
      filter_data = gland_coor_conf[indices_to_keep]
      sort_filter_data = filter_data[filter_data[:, 0].argsort()]

      self.gland_images.append(self.gland_images[0].copy())
      self.gland_images.append(self.image.copy())
      for i in range(len(sort_filter_data)):
         self.gland_images[2] = cv2.circle( self.gland_images[2], (int(sort_filter_data[i,0]),int(sort_filter_data[i,1])), 15, (0,255,255), 2)
         self.gland_images[3] = cv2.circle( self.gland_images[3], (int(sort_filter_data[i,0]),int(sort_filter_data[i,1])), 15, (0,255,255), 2)

      median_all = np.median(sort_filter_data, axis=0)
      median_x = self.image.shape[1]/2 #median_all[0]
      median_y = median_all[1]
      distance_to_median = (sort_filter_data[:, 0] - median_x) ** 2 + (sort_filter_data[:, 1] - median_y) ** 2
      idx_min = np.argmin(distance_to_median)

      gland_list = [sort_filter_data[idx_min]]

      self.gland_images[2] = cv2.circle( self.gland_images[2], (int(sort_filter_data[idx_min,0]),int(sort_filter_data[idx_min,1])), 30, (255,255,255), 2)

      current_y = sort_filter_data[idx_min][0]
      current_x = sort_filter_data[idx_min][1]
      right = sort_filter_data[idx_min+1::,:]
      for i in range(right.shape[0]):
        distance = ((right[:,0] - current_y)**2 + (right[:,1] - current_x)**2)**0.5
        next_idx =  np.argmin(distance)
        print('DEG:',math.atan2(abs(right[next_idx,1] - current_x),abs(right[next_idx,0] - current_y))/math.pi * 180)
        print('dx:',abs(right[next_idx,0] - current_y)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]))
        print('dy:',abs(right[next_idx,1] - current_x)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]))
        if math.atan2(abs(right[next_idx,1] - current_x),abs(right[next_idx,0] - current_y)) > math.pi*0.45:
          right = np.delete(right, next_idx, 0)
        elif (abs(right[next_idx,1] - current_x)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]) > 0.18):
          right = np.delete(right, next_idx, 0)
        else:
          gland_list.append(right[next_idx])
          current_y = right[next_idx][0]
          current_x = right[next_idx][1]
          right = right[next_idx+1::,:]
        if len(right) == 0:
          break

      print('-------------------------------------')

      current_y = sort_filter_data[idx_min][0]
      current_x = sort_filter_data[idx_min][1]
      left = sort_filter_data[idx_min-1::-1,:]
      for i in range(left.shape[0]):
        distance = ((left[:,0] - current_y)**2 + (left[:,1] - current_x)**2)**0.5
        next_idx =  np.argmin(distance)
        print('DEG:',math.atan2(abs(left[next_idx,1] - current_x),abs(left[next_idx,0] - current_y))/math.pi * 180)
        print('dx:',abs(left[next_idx,0] - current_y)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]))
        print('dy:',abs(left[next_idx,1] - current_x)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]))
        if (math.atan2(abs(left[next_idx,1] - current_x),abs(left[next_idx,0] - current_y)) > math.pi*0.45):
          left = np.delete(left, next_idx, 0)
        elif (abs(left[next_idx,1] - current_x)/(self.eyelid_crop_idx[1] - self.eyelid_crop_idx[0]) > 0.18):
          left = np.delete(left, next_idx, 0)
        else:
          gland_list.insert(0,left[next_idx])
          current_y = left[next_idx][0]
          current_x = left[next_idx][1]
          left = left[next_idx+1::,:]
        if len(left) == 0:
          break

      self.gland_list = np.vstack(gland_list)

      self.gland_images.append(self.gland_images[2].copy())
      for i in range(self.gland_list.shape[0]-1):
        self.gland_images[4] = cv2.line(self.gland_images[4], (int(self.gland_list[i,0]),int(self.gland_list[i,1])), (int(self.gland_list[i+1,0]),int(self.gland_list[i+1,1])), (255,255,0), 2)

    def find_abnormality_score(self):
      theta = []
      distance = []
      for i in range(self.gland_list.shape[0] - 1):
        temp = math.degrees(math.atan2(self.gland_list[i+1, 1] - self.gland_list[i, 1], self.gland_list[i+1, 0] - self.gland_list[i, 0]))
        d = ((self.gland_list[i+1, 1] - self.gland_list[i, 1])**2 + 1.5*(self.gland_list[i+1, 0] - self.gland_list[i, 0])**2)**0.5
        theta.append(temp)
        distance.append(d)

      self.theta = np.array(theta)
      self.abnormality_score[0] = np.sum(np.abs(self.theta[1::]-self.theta[:-1:]))/(len(self.theta)-1)
      weight = np.maximum(distance[1::], distance[:-1:])
      weight = (weight-np.mean(weight))/np.std(weight)
      weight[weight < 0] = 0;
      weight[np.abs(self.theta[1::]-self.theta[:-1:]) < 20] = 0;
      weight = (2*weight+1)**(-1)
      self.abnormality_score[1] = np.sum((np.abs(self.theta[1::]-self.theta[:-1:])*weight)/np.sum(weight))

      print('Theta:', self.theta)
      print('Distance:', distance)
      print('Weight:', weight)
    #   print(self.abnormality_score)
      return self.abnormality_score[1]


# def abnormality_grade(url):
    
@app.route('/predict', methods=['POST'])
def abnormality_grade_route():
    try:
        eyelid_detector = './eyelid.pt'
        gland_detector = './gland.pt'
        data = request.get_json()
        image_urls = data['body']['url']
        analyzer = MeibomianGlandAnalyzer(image = image_urls,
                                        eyelid_detector = eyelid_detector,
                                        gland_detector = gland_detector,
                                        eyelid_detector_conf = 0.1,
                                        gland_detector_conf = 0.2)
        analyzer.detect_eyelid()
        analyzer.detect_gland()
        
        score = analyzer.find_abnormality_score()
        result = {'rms_values': score}
        print(result)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
