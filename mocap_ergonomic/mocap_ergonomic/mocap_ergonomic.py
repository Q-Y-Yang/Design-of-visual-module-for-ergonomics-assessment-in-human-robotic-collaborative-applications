#!/usr/bin/env python3
#!coding=utf-8

import cv2
import sys
import torch
import numpy as np
import pickle 
from torchvision.transforms import Normalize

sys.path.insert(0, '/home/student/frankmocap')
sys.path.insert(0, '/home/student/frankmocap/detectors/body_pose_estimator')

from bodymocap.models import hmr, SMPL, SMPLX
from bodymocap import constants
from bodymocap.utils.imutils import crop, crop_bboxInfo, process_image_bbox, process_image_keypoints, bbox_from_keypoints
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm
import mocap_utils.geometry_utils as gu
from detectors.body_pose_estimator.val import normalize, pad_width
from detectors.body_pose_estimator.pose2d_models.with_mobilenet import PoseEstimationWithMobileNet
from detectors.body_pose_estimator.modules.load_state import load_state
from detectors.body_pose_estimator.modules.pose import Pose, track_poses
from detectors.body_pose_estimator.modules.keypoints import extract_keypoints, group_keypoints
from renderer.visualizer import Visualizer
import mocap_utils.demo_utils as demo_utils
from renderer.viewer2D import ImShow

import os, shutil
import os.path as osp
from torchvision.transforms import transforms

from handmocap.hand_modules.test_options import TestOptions
from handmocap.hand_modules.h3dw_model import H3DWModel
from mocap_utils.coordconv import convert_smpl_to_bbox, convert_bbox_to_oriIm

from integration import copy_and_paste


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Int64
from cv_bridge import CvBridge, CvBridgeError
from . import eva3d
import time

sys.path.insert(0, '/home/student/frankmocap/handmocap')
from hand_bbox_detector import HandBboxDetector

import argparse

import pdb



class BodyMocap(Node):
    
    def __init__(self, body_regressor_checkpoint, hand_regressor_checkpoint, smpl_dir, device=torch.device('cuda'), use_smplx=True):
        super().__init__('BodyMocap')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print("Loading Body Pose Estimator")
        self.__load_body_estimator()
        self.visualizer = Visualizer('opengl')
        self.frame_id = 0	#count frames

        parser = argparse.ArgumentParser()
        parser.add_argument("--rot90", default=False, type= bool, help="clockwise rotate 90 degrees")
        #parser.add_argument("--camera_topic", default="/logi_c922_2/image_rect_color", help="choose a topic as input image")
        parser.add_argument("--body_only", default=False, type= bool, help="detect only body and save its result")
        parser.add_argument("--result_path", default="/home/student/result/", help="choose a topic as input image")
        parser.add_argument("--save_result", default=False, help="save result or not")
        args = parser.parse_args()
        self.rot90 = args.rot90
        #self.camera_topic = args.camera_topic
        self.body_only = args.body_only
        self.result_path = args.result_path
        self.save_result = args.save_result
        self.load = [0,0]
        self.angle_leg = 0
        self.angle_trunk = 0
        self.start = 0
        self.angles =  np.empty((1,20),dtype = float)
        self.body_side =  np.empty((25,3),dtype = float)
        # Load parametric model (SMPLX or SMPL)
        if use_smplx:
            smplModelPath = smpl_dir + '/SMPLX_NEUTRAL.pkl'
            self.smpl = SMPLX(smpl_dir,
                    batch_size=1,
                    num_betas = 10,
                    use_pca = False,
                    create_transl=False).to(self.device)
            self.use_smplx = True
        else:
            smplModelPath = smpl_dir + '/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl'
            self.smpl = SMPL(smplModelPath, batch_size=1, create_transl=False).to(self.device)
            self.use_smplx = False
            
        #Load pre-trained neural network 
        SMPL_MEAN_PARAMS = '/home/student/frankmocap/extra_data/body_module/data_from_spin/smpl_mean_params.npz'
        self.model_regressor = hmr(SMPL_MEAN_PARAMS).to(self.device)
        body_checkpoint = torch.load(body_regressor_checkpoint)
        self.model_regressor.load_state_dict(body_checkpoint['model'], strict=False)
        self.model_regressor.eval()

       #hand module init
        
        transform_list = [ transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        self.normalize_transform = transforms.Compose(transform_list)

        #Load Hand network 
        self.opt = TestOptions().parse([])

        #Default options
        self.opt.single_branch = True
        self.opt.main_encoder = "resnet50"
        # self.opt.data_root = "/home/hjoo/dropbox/hand_yu/data/"
        self.opt.model_root = "/home/student/frankmocap/extra_data"
        self.opt.smplx_model_file = os.path.join(smpl_dir,'SMPLX_NEUTRAL.pkl')
      
        self.opt.batchSize = 1
        self.opt.phase = "test"
        self.opt.nThreads = 0
        self.opt.which_epoch = -1
        self.opt.checkpoint_path = hand_regressor_checkpoint

        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.process_rank = -1

        # self.opt.which_epoch = str(epoch)
        self.hand_model_regressor = H3DWModel(self.opt)
        # if there is no specified checkpoint, then skip
        assert self.hand_model_regressor.success_load, "Specificed checkpoints does not exists: {}".format(self.opt.checkpoint_path)
        self.hand_model_regressor.eval()

        self.hand_bbox_detector = HandBboxDetector('third_view', self.device)

 		#subscriber and publisher initialization
		#input subscriber
        self.br = CvBridge()
        self.subscription_img = self.create_subscription(Image, '/side_img', self.callback_side,10)
        self.subscription_img = self.create_subscription(Image, '/front_img', self.callback_front,10)
		
		#output publisher
        self.publisher_pose = self.create_publisher(Image,'/pose',10)	#images with keypoints annotation
        #self.publisher_keypoints = self.create_publisher(Float32MultiArray,'/keypoints',10)	#keypoints coordinates
        self.publisher_risk = self.create_publisher(Int64,'/risk',10)	#risk level
        self.publisher_angles = self.create_publisher(Float32MultiArray,'/angles',10)

    def callback_side(self, data):
        self.start = time.time()
        try:
            img_original = self.br.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        self.frame_id = self.frame_id + 1
        pose, body_bbox_list = self.detect_body_pose(img_original)

        pred_body_list = self.body_regress(img_original, body_bbox_list)
       
        self.body_side=np.array(pred_body_list[0]['pred_joints_img'][1:25,:])

        #render
        pred_mesh_list = demo_utils.extract_mesh_from_output(pred_body_list)
        imgside_render = self.visualizer.visualize(img_original, pred_mesh_list = pred_mesh_list,body_bbox_list = body_bbox_list)
        imgside_render = imgside_render.astype(np.uint8)
        self.publisher_pose.publish(self.br.cv2_to_imgmsg(imgside_render))

        #save result
        if self.save_result is True:
            cv2.imwrite(str(self.result_path) +'side'+str(self.frame_id)+'.jpg',imgside_render)

    def callback_front(self,data):       
        try:
            if self.rot90 is True:
                img_original = cv2.flip(cv2.transpose(self.br.imgmsg_to_cv2(data, "bgr8")), 0)
            else:
                img_original = self.br.imgmsg_to_cv2(data, "bgr8")
          
            self.frame_id = self.frame_id + 1
            img_shape = [img_original.shape[0],img_original.shape[1]]
            h,w = img_shape
        except CvBridgeError as e:
            print(e)
        pose, body_bbox_list = self.detect_body_pose(img_original)

        pred_body_list = self.body_regress(img_original, body_bbox_list)

        risk = Int64()
        angles = Float32MultiArray()

        #hand detection and integration
        if self.body_only is False:    
           #hand_bbox_list = self.get_hand_bboxes(pred_body_list, img_shape)
            body_pose_list, body_bbox_list, hand_bbox_list, raw_hand_bboxes = self.hand_bbox_detector.detect_hand_bbox(img_original)
            
            pred_hand_list = self.hand_regress(img_original, hand_bbox_list, add_margin=False)
        
            integrate_output_list = self.integration_copy_paste(pred_body_list, pred_hand_list, self.smpl, img_shape)
            body=np.array(integrate_output_list[0]['pred_body_joints_img'][0:25,:])
            rhand=np.array(integrate_output_list[0]['pred_rhand_joints_img'][0:21,:])
            lhand=np.array(integrate_output_list[0]['pred_lhand_joints_img'][0:21,:])
            hands=np.vstack((lhand,rhand))
            whole_body=np.vstack((body,hands))
            #print(self.angle_leg)
            risklevel, self.angles = eva3d.scoring(whole_body, self.body_side, self.load)
            #print(risklevel, main_angles)

            end = time.time()
            fps = 1 / (end - self.start)
            print('FPS:'+str(fps))
            risk.data = risklevel.item()
            angles.data = tuple(self.angles)

            #output
            self.publisher_risk.publish(risk)
            self.publisher_angles.publish(angles)

            #render and visulization
            pred_mesh_list = demo_utils.extract_mesh_from_output(integrate_output_list)
            img_render = self.visualizer.visualize(img_original, pred_mesh_list = pred_mesh_list,body_bbox_list = body_bbox_list,hand_bbox_list = hand_bbox_list)
            img_render = img_render.astype(np.uint8)
            self.publisher_pose.publish(self.br.cv2_to_imgmsg(img_render))

            #save result
            if self.save_result is True:
                cv2.imwrite(str(self.result_path) +str(self.frame_id)+'.jpg',img_render)
                with open(str(self.result_path)+ 'results.txt', 'a') as file_handle:
                        file_handle.write('\nframe_id:')
                        file_handle.write(str(self.frame_id))
                        file_handle.write('\nrisk_level:')
                        file_handle.write(str(risklevel))
                        file_handle.write('\nmain_angles:')
                        file_handle.write(str(main_angles))

               

    def body_regress(self, img_original, body_bbox_list):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                body_bbox: bounding box around the target: (minX, minY, width, height)
            outputs:
                pred_vertices_img:
                pred_joints_vis_img:
                pred_rotmat
                pred_betas
                pred_camera
                bbox: [bbr[0], bbr[1],bbr[0]+bbr[2], bbr[1]+bbr[3]])
                bboxTopLeft:  bbox top left (redundant)
                boxScale_o2n: bbox scaling factor (redundant) 
        """
        pred_output_list = list()

        for body_bbox in body_bbox_list:
            img, norm_img, boxScale_o2n, bboxTopLeft, bbox = process_image_bbox(
                img_original, body_bbox, input_res=constants.IMG_RES)
            bboxTopLeft = np.array(bboxTopLeft)

            # bboxTopLeft = bbox['bboxXYWH'][:2]
            if img is None:
                pred_output_list.append(None)
                continue

            with torch.no_grad():
                # model forward
                pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))

                #Convert rot_mat to aa since hands are always in aa
                # pred_aa = rotmat3x3_to_angle_axis(pred_rotmat)
                pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
                pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
                smpl_output = self.smpl(
                    betas=pred_betas, 
                    body_pose=pred_aa[:,3:],
                    global_orient=pred_aa[:,:3], 
                    pose2rot=True)
                pred_vertices = smpl_output.vertices
                pred_joints_3d = smpl_output.joints

                pred_vertices = pred_vertices[0].cpu().numpy()

                pred_camera = pred_camera.cpu().numpy().ravel()
                camScale = pred_camera[0] # *1.15
                camTrans = pred_camera[1:]

                pred_output = dict()
                # Convert mesh to original image space (X,Y are aligned to image)
                # 1. SMPL -> 2D bbox
                # 2. 2D bbox -> original 2D image
                pred_vertices_bbox = convert_smpl_to_bbox(pred_vertices, camScale, camTrans)
                pred_vertices_img = convert_bbox_to_oriIm(
                    pred_vertices_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])

                # Convert joint to original image space (X,Y are aligned to image)
                pred_joints_3d = pred_joints_3d[0].cpu().numpy() # (1,49,3)
                pred_joints_vis = pred_joints_3d[:,:3]  # (49,3)
    
                pred_joints_vis_bbox = convert_smpl_to_bbox(pred_joints_vis, camScale, camTrans) 
                pred_joints_vis_img = convert_bbox_to_oriIm(
                    pred_joints_vis_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0]) 

                # Output
                pred_output['img_cropped'] = img[:, :, ::-1]
                pred_output['pred_vertices_smpl'] = smpl_output.vertices[0].cpu().numpy() # SMPL vertex in original smpl space
                pred_output['pred_vertices_img'] = pred_vertices_img # SMPL vertex in image space
                pred_output['pred_joints_img'] = pred_joints_vis_img # SMPL joints in image space

                pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat.detach().cpu()[0])
                pred_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72)
                pred_body_pose = pred_output['pred_body_pose']
                pred_output['pred_rotmat'] = pred_rotmat.detach().cpu().numpy() # (1, 24, 3, 3)
                pred_output['pred_betas'] = pred_betas.detach().cpu().numpy() # (1, 10)

                pred_output['pred_camera'] = pred_camera
                pred_output['bbox_top_left'] = bboxTopLeft
                pred_output['bbox_scale_ratio'] = boxScale_o2n
                pred_output['faces'] = self.smpl.faces

                if self.use_smplx:
                    img_center = np.array((img_original.shape[1], img_original.shape[0]) ) * 0.5
                    # right hand
                    pred_joints = smpl_output.right_hand_joints[0].cpu().numpy()     
                    pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                    pred_joints_img = convert_bbox_to_oriIm(
                        pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                    pred_output['right_hand_joints_img_coord'] = pred_joints_img
                    # left hand 
                    pred_joints = smpl_output.left_hand_joints[0].cpu().numpy()
                    pred_joints_bbox = convert_smpl_to_bbox(pred_joints, camScale, camTrans)
                    pred_joints_img = convert_bbox_to_oriIm(
                        pred_joints_bbox, boxScale_o2n, bboxTopLeft, img_original.shape[1], img_original.shape[0])
                    pred_output['left_hand_joints_img_coord'] = pred_joints_img
                
                pred_output_list.append(pred_output)
            if self.body_only is True:
                with open('/home/student/bodyonly_side/' + 'results.txt', 'a') as file_handle:
    	                file_handle.write('\nframe_id:')
    	                file_handle.write(str(self.frame_id))
    	                file_handle.write('\npred_body_pose:')
    	                file_handle.write(str(pred_body_pose))
    	                file_handle.write('\npred_joints_vis_img:')
    	                file_handle.write(str(pred_joints_vis_img))
		    #save images(only body_module has visualizer)
		    # extract mesh for rendering (vertices in image space and faces) from pred_output_list
                pred_mesh_list = demo_utils.extract_mesh_from_output(pred_output_list)
		     # visualization
                res_img = self.visualizer.visualize(img_original,pred_mesh_list = pred_mesh_list, body_bbox_list = body_bbox_list)

		    # show result in the screen
		   
                res_img = res_img.astype(np.uint8)
		   # ImShow(res_img)
		    
                cv2.imwrite('/home/student/bodyonly_side/' +str(self.frame_id)+'.jpg',res_img)

        return pred_output_list

    def __pad_and_resize(self, img, hand_bbox, add_margin, final_size=224):
        ori_height, ori_width = img.shape[:2]
        min_x, min_y = hand_bbox[:2].astype(np.int32)
        width, height = hand_bbox[2:].astype(np.int32)
        max_x = min_x + width
        max_y = min_y + height

        if width > height:
            margin = (width-height) // 2
            min_y = max(min_y-margin, 0)
            max_y = min(max_y+margin, ori_height)
        else:
            margin = (height-width) // 2
            min_x = max(min_x-margin, 0)
            max_x = min(max_x+margin, ori_width)
        
        # add additional margin
        if add_margin:
            margin = int(0.3 * (max_y-min_y)) # if use loose crop, change 0.03 to 0.1
            min_y = max(min_y-margin, 0)
            max_y = min(max_y+margin, ori_height)
            min_x = max(min_x-margin, 0)
            max_x = min(max_x+margin, ori_width)

        img_cropped = img[int(min_y):int(max_y), int(min_x):int(max_x), :]
        new_size = max(max_x-min_x, max_y-min_y)
        new_img = np.zeros((new_size, new_size, 3), dtype=np.uint8)
        # new_img = np.zeros((new_size, new_size, 3))
        new_img[:(max_y-min_y), :(max_x-min_x), :] = img_cropped
        bbox_processed = (min_x, min_y, max_x, max_y)

        # resize to 224 * 224
        new_img = cv2.resize(new_img, (final_size, final_size))

        ratio = final_size / new_size
        return new_img, ratio, (min_x, min_y, max_x-min_x, max_y-min_y)

    def __process_hand_bbox(self, raw_image, hand_bbox, hand_type, add_margin=True):
        """
        args: 
            original image, 
            bbox: (x0, y0, w, h)
            hand_type ("left_hand" or "right_hand")
            add_margin: If the input hand bbox is a tight bbox, then set this value to True, else False
        output:
            img_cropped: 224x224 cropped image (original colorvalues 0-255)
            norm_img: 224x224 cropped image (normalized color values)
            bbox_scale_ratio: scale factor to convert from original to cropped
            bbox_top_left_origin: top_left corner point in original image cooridate
        """
        # print("hand_type", hand_type)

        assert hand_type in ['left_hand', 'right_hand']
        img_cropped, bbox_scale_ratio, bbox_processed = \
            self.__pad_and_resize(raw_image, hand_bbox, add_margin)
        
        #horizontal Flip to make it as right hand
        if hand_type=='left_hand':
            img_cropped = np.ascontiguousarray(img_cropped[:, ::-1,:], img_cropped.dtype) 
        else:
            assert hand_type == 'right_hand'

        # img normalize
        norm_img = self.normalize_transform(img_cropped).float()
        # return
        return img_cropped, norm_img, bbox_scale_ratio, bbox_processed

    def hand_regress(self, img_original, hand_bbox_list, add_margin=False):
        """
            args: 
                img_original: original raw image (BGR order by using cv2.imread)
                hand_bbox_list: [
                    dict(
                        left_hand = [x0, y0, w, h] or None
                        right_hand = [x0, y0, w, h] or None
                    )
                    ...
                ]
                add_margin: whether to do add_margin given the hand bbox
            outputs:
                To be filled
            Note: 
                Output element can be None. This is to keep the same output size with input bbox
        """
        hand_pred_output_list = list()
        hand_bbox_list_processed = list()

        for hand_bboxes in hand_bbox_list:

            if hand_bboxes is None:     # Should keep the same size with bbox size
                hand_pred_output_list.append(None)
                hand_bbox_list_processed.append(None)
                continue

            hand_pred_output = dict(
                left_hand = None,
                right_hand = None
            )
            hand_bboxes_processed = dict(
                left_hand = None,
                right_hand = None
            )

            for hand_type in hand_bboxes:
                bbox = hand_bboxes[hand_type]
                
                if bbox is None: 
                    continue
                else:
                    img_cropped, norm_img, bbox_scale_ratio, bbox_processed = \
                        self.__process_hand_bbox(img_original, hand_bboxes[hand_type], hand_type, add_margin)
                    hand_bboxes_processed[hand_type] = bbox_processed

                    with torch.no_grad():
                        # pred_rotmat, pred_betas, pred_camera = self.model_regressor(norm_img.to(self.device))
                        self.hand_model_regressor.set_input_imgonly({'img': norm_img.unsqueeze(0)})
                        self.hand_model_regressor.test()
                        pred_res = self.hand_model_regressor.get_pred_result()

                        ##Output
                        cam = pred_res['cams'][0, :]  #scale, tranX, tranY
                        pred_verts_origin = pred_res['pred_verts'][0]
                        faces = self.hand_model_regressor.right_hand_faces_local
                        pred_pose = pred_res['pred_pose_params'].copy()
                        pred_joints = pred_res['pred_joints_3d'].copy()[0]

                        if hand_type == 'left_hand':
                            cam[1] *= -1
                            pred_verts_origin[:, 0] *= -1
                            faces = faces[:, ::-1]
                            pred_pose[:, 1::3] *= -1
                            pred_pose[:, 2::3] *= -1
                            pred_joints[:, 0] *= -1

                        hand_pred_output[hand_type] = dict()
                        hand_pred_output[hand_type]['pred_vertices_smpl'] = pred_verts_origin # SMPL-X hand vertex in bbox space
                        hand_pred_output[hand_type]['pred_joints_smpl'] = pred_joints
                        hand_pred_output[hand_type]['faces'] = faces

                        hand_pred_output[hand_type]['bbox_scale_ratio'] = bbox_scale_ratio
                        hand_pred_output[hand_type]['bbox_top_left'] = np.array(bbox_processed[:2])
                        hand_pred_output[hand_type]['pred_camera'] = cam
                        hand_pred_output[hand_type]['img_cropped'] = img_cropped

                        # pred hand pose & shape params & hand joints 3d
                        hand_pred_output[hand_type]['pred_hand_pose'] = pred_pose
                        hand_pred_output[hand_type]['pred_hand_betas'] = pred_res['pred_shape_params'] # (1, 10)
                        
                        #Convert vertices into bbox & image space
                        cam_scale = cam[0]
                        cam_trans = cam[1:]
                        vert_smplcoord = pred_verts_origin.copy()
                        joints_smplcoord = pred_joints.copy()
                        
                        vert_bboxcoord = convert_smpl_to_bbox(
                            vert_smplcoord, cam_scale, cam_trans, bAppTransFirst=True) # SMPL space -> bbox space
                        joints_bboxcoord = convert_smpl_to_bbox(
                            joints_smplcoord, cam_scale, cam_trans, bAppTransFirst=True) # SMPL space -> bbox space

                        hand_boxScale_o2n = hand_pred_output[hand_type]['bbox_scale_ratio']
                        hand_bboxTopLeft = hand_pred_output[hand_type]['bbox_top_left']

                        vert_imgcoord = convert_bbox_to_oriIm(
                                vert_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 
                                img_original.shape[1], img_original.shape[0]) 
                        hand_pred_output[hand_type]['pred_vertices_img'] = vert_imgcoord

                        joints_imgcoord = convert_bbox_to_oriIm(
                                joints_bboxcoord, hand_boxScale_o2n, hand_bboxTopLeft, 
                                img_original.shape[1], img_original.shape[0]) 
                        hand_pred_output[hand_type]['pred_joints_img'] = joints_imgcoord
                        
            hand_pred_output_list.append(hand_pred_output)
            hand_bbox_list_processed.append(hand_bboxes_processed)
          
        assert len(hand_bbox_list_processed) == len(hand_bbox_list)
        return hand_pred_output_list
    

    def get_hand_bboxes(self, pred_body_list, img_shape):
        """
            args: 
                pred_body_list: output of body regresion
                img_shape: img_height, img_width
            outputs:
                hand_bbox_list: 
        """
        hand_bbox_list = list()
        for pred_body in pred_body_list:
            hand_bbox = dict(
                left_hand = None,
                right_hand = None
            )
            if pred_body is None:
                hand_bbox_list.append(hand_bbox)
            else:
                for hand_type in hand_bbox:
                    key = f'{hand_type}_joints_img_coord'
                    pred_joints_vis_img = pred_body[key]

                    if pred_joints_vis_img is not None:
                        # get initial bbox
                        x0, x1 = np.min(pred_joints_vis_img[:, 0]), np.max(pred_joints_vis_img[:, 0])
                        y0, y1 = np.min(pred_joints_vis_img[:, 1]), np.max(pred_joints_vis_img[:, 1])
                        width, height = x1-x0, y1-y0
                        # extend the obtained bbox
                        margin = int(max(height, width) * 0.2)
                        img_height, img_width = img_shape
                        x0 = max(x0 - margin, 0)
                        y0 = max(y0 - margin, 0)
                        x1 = min(x1 + margin, img_width)
                        y1 = min(y1 + margin, img_height)
                        # result bbox in (x0, y0, w, h) format
                        hand_bbox[hand_type] = np.array([x0, y0, x1-x0, y1-y0]) # in (x, y, w, h ) format

                hand_bbox_list.append(hand_bbox)

        return hand_bbox_list



    

    def __load_body_estimator(self):
        net = PoseEstimationWithMobileNet()
        pose2d_checkpoint = "/home/student/frankmocap/extra_data/body_module/body_pose_estimator/checkpoint_iter_370000.pth"
        checkpoint = torch.load(pose2d_checkpoint, map_location='cpu')
        load_state(net, checkpoint)
        net = net.eval()
        net = net.cuda()
        self.model = net

    def __infer_fast(self, img, input_height_size, stride, upsample_ratio, 
        cpu=False, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
        height, width, _ = img.shape
        scale = input_height_size / height

        scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        scaled_img = normalize(scaled_img, img_mean, img_scale)
        min_dims = [input_height_size, max(scaled_img.shape[1], input_height_size)]
        padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

        tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
        if not cpu:
            tensor_img = tensor_img.cuda()

        stages_output = self.model(tensor_img)

        stage2_heatmaps = stages_output[-2]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        stage2_pafs = stages_output[-1]
        pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
        pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

        return heatmaps, pafs, scale, pad
    
    def detect_body_pose(self, img):
        """
        Output:
            current_bbox: BBOX_XYWH
        """
        stride = 8
        upsample_ratio = 4
        orig_img = img.copy()

        # forward
        heatmaps, pafs, scale, pad = self.__infer_fast(img, 
            input_height_size=256, stride=stride, upsample_ratio=upsample_ratio)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        num_keypoints = Pose.num_kpts
        for kpt_idx in range(num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
        
        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs, demo=True)
        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
        
        '''
        # print(len(pose_entries))
        if len(pose_entries)>1:
            pose_entries = pose_entries[:1]
            print("We only support one person currently")
            # assert len(pose_entries) == 1, "We only support one person currently"
        '''

        current_poses, current_bbox = list(), list()
        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
            for kpt_id in range(num_keypoints):
                if pose_entries[n][kpt_id] != -1.0:  # keypoint was found
                    pose_keypoints[kpt_id, 0] = int(all_keypoints[int(pose_entries[n][kpt_id]), 0])
                    pose_keypoints[kpt_id, 1] = int(all_keypoints[int(pose_entries[n][kpt_id]), 1])
            pose = Pose(pose_keypoints, pose_entries[n][18]) 
            current_poses.append(pose.keypoints)
            current_bbox.append(np.array(pose.bbox))

        # enlarge the bbox
        for i, bbox in enumerate(current_bbox):
            x, y, w, h = bbox
            margin = 0.05
            x_margin = int(w * margin)
            y_margin = int(h * margin)
            x0 = max(x-x_margin, 0)
            y0 = max(y-y_margin, 0)
            x1 = min(x+w+x_margin, orig_img.shape[1])
            y1 = min(y+h+y_margin, orig_img.shape[0])
            current_bbox[i] = np.array((x0, y0, x1-x0, y1-y0)).astype(np.int32)

        return current_poses, current_bbox





    def get_kinematic_map(self, smplx_model, dst_idx):
        cur = dst_idx
        kine_map = dict()
        while cur>=0:
            parent = int(smplx_model.parents[cur])
            if cur != dst_idx: # skip the dst_idx itself
                kine_map[parent] = cur
            cur = parent
        return kine_map


    def __transfer_rot(self, body_pose_rotmat, part_rotmat, kinematic_map, transfer_type):

        rotmat= body_pose_rotmat[0] 
        parent_id = 0
        while parent_id in kinematic_map:
            child_id = kinematic_map[parent_id]
            local_rotmat = body_pose_rotmat[child_id]
            rotmat = torch.matmul(rotmat, local_rotmat)
            parent_id = child_id

        if transfer_type == 'g2l':
            part_rot_new = torch.matmul(rotmat.T, part_rotmat)
        else:
            assert transfer_type == 'l2g'
            part_rot_new = torch.matmul(rotmat, part_rotmat)

        return part_rot_new


    def transfer_rotation(self,
        smplx_model, body_pose, part_rot, part_idx, 
        transfer_type="g2l", result_format="rotmat"):

        assert transfer_type in ["g2l", "l2g"]
        assert result_format in ['rotmat', 'aa']

        assert type(body_pose) == type(part_rot)
        return_np = False

        if isinstance(body_pose, np.ndarray):
            body_pose = torch.from_numpy(body_pose)
            return_np = True
        
        if isinstance(part_rot, np.ndarray):
            part_rot = torch.from_numpy(part_rot)
            return_np = True

        if body_pose.dim() == 2:
            # aa
            assert body_pose.size(0) == 1 and body_pose.size(1) in [66, 72]
            body_pose_rotmat = gu.angle_axis_to_rotation_matrix(body_pose.view(22, 3)).clone()
        else:
            # rotmat
            assert body_pose.dim() == 4
            assert body_pose.size(0) == 1 and body_pose.size(1) in [22, 24]
            assert body_pose.size(2) == 3 and body_pose.size(3) == 3
            body_pose_rotmat = body_pose[0].clone()

        if part_rot.dim() == 2:
            # aa
            assert part_rot.size(0) == 1 and part_rot.size(1) == 3
            part_rotmat = gu.angle_axis_to_rotation_matrix(part_rot)[0,:3,:3].clone()
        else:
            # rotmat
            assert part_rot.dim() == 3
            assert part_rot.size(0) == 1 and part_rot.size(1) == 3 and part_rot.size(2) == 3
            part_rotmat = part_rot[0,:3,:3].clone()

        kinematic_map = self.get_kinematic_map(smplx_model, part_idx)
        part_rot_trans = self.__transfer_rot(
            body_pose_rotmat, part_rotmat, kinematic_map, transfer_type)

        if result_format == 'rotmat':    
            return_value = part_rot_trans
        else:
            part_rot_aa = gu.rotation_matrix_to_angle_axis(part_rot_trans)
            return_value = part_rot_aa
        if return_np:
            return_value = return_value.numpy()
        return return_value


    def integration_copy_paste(self, pred_body_list, pred_hand_list, smplx_model, image_shape):
        integral_output_list = list()
        for i in range(len(pred_body_list)):
            body_info = pred_body_list[i]
        for j in range(len(pred_hand_list)):
            hand_info = pred_hand_list[j]
            if body_info is None:
                integral_output_list.append(None)
                continue
        
            # copy and paste 
            pred_betas = torch.from_numpy(body_info['pred_betas']).cuda()
            pred_rotmat = torch.from_numpy(body_info['pred_rotmat']).cuda()

            # integrate right hand pose
            hand_output = dict()
            if hand_info is not None and hand_info['right_hand'] is not None:
                right_hand_pose = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:, 3:]).cuda()
                right_hand_global_orient = torch.from_numpy(hand_info['right_hand']['pred_hand_pose'][:,: 3]).cuda()
                right_hand_local_orient = self.transfer_rotation(
                    smplx_model, pred_rotmat, right_hand_global_orient, 21)
                pred_rotmat[0, 21] = right_hand_local_orient
            else:
                right_hand_pose = torch.from_numpy(np.zeros( (1,45) , dtype= np.float32)).cuda()
                right_hand_global_orient = None
                right_hand_local_orient = None

            # integrate left hand pose
            if hand_info is not None and hand_info['left_hand'] is not None:
                left_hand_pose = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, 3:]).cuda()
                left_hand_global_orient = torch.from_numpy(hand_info['left_hand']['pred_hand_pose'][:, :3]).cuda()
                left_hand_local_orient = self.transfer_rotation(
                    smplx_model, pred_rotmat, left_hand_global_orient, 20)
                pred_rotmat[0, 20] = left_hand_local_orient
            else:
                left_hand_pose = torch.from_numpy(np.zeros((1,45), dtype= np.float32)).cuda()
                left_hand_global_orient = None
                left_hand_local_orient = None

            pred_aa = gu.rotation_matrix_to_angle_axis(pred_rotmat).cuda()
            pred_aa = pred_aa.reshape(pred_aa.shape[0], 72)
            smplx_output = smplx_model(
                betas = pred_betas, 
                body_pose = pred_aa[:,3:], 
                global_orient = pred_aa[:,:3],
                right_hand_pose = right_hand_pose, 
                left_hand_pose= left_hand_pose,
                pose2rot = True)

            pred_vertices = smplx_output.vertices
            pred_vertices = pred_vertices[0].detach().cpu().numpy()   
            pred_body_joints = smplx_output.joints
            pred_body_joints = pred_body_joints[0].detach().cpu().numpy()   
            pred_lhand_joints = smplx_output.left_hand_joints
            pred_lhand_joints = pred_lhand_joints[0].detach().cpu().numpy()
            pred_rhand_joints = smplx_output.right_hand_joints
            pred_rhand_joints = pred_rhand_joints[0].detach().cpu().numpy()

            camScale = body_info['pred_camera'][0]
            camTrans = body_info['pred_camera'][1:]
            bbox_top_left = body_info['bbox_top_left']
            bbox_scale_ratio = body_info['bbox_scale_ratio']

                # Convert joint to original image space (X,Y are aligned to image)
         
 
            integral_output = dict()
        
            integral_output['pred_vertices_smpl'] = pred_vertices
            integral_output['faces'] = smplx_model.faces
            integral_output['bbox_scale_ratio'] = bbox_scale_ratio
            integral_output['bbox_top_left'] = bbox_top_left
            integral_output['pred_camera'] = body_info['pred_camera']

            pred_aa_tensor = gu.rotation_matrix_to_angle_axis(pred_rotmat.detach().cpu()[0])
            integral_output['pred_body_pose'] = pred_aa_tensor.cpu().numpy().reshape(1, 72)
            integral_output['pred_betas'] = pred_betas.detach().cpu().numpy()

            # convert mesh to original image space (X,Y are aligned to image)
            pred_vertices_bbox = convert_smpl_to_bbox(
                pred_vertices, camScale, camTrans)
            pred_vertices_img = convert_bbox_to_oriIm(
                pred_vertices_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
            integral_output['pred_vertices_img'] = pred_vertices_img

            # convert joints to original image space (X, Y are aligned to image)
            pred_body_joints_bbox = convert_smpl_to_bbox(
                pred_body_joints, camScale, camTrans)
            pred_body_joints_img = convert_bbox_to_oriIm(
                pred_body_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
            integral_output['pred_body_joints_img'] = pred_body_joints_img
            #self.body_3d = pred_body_joints_img

            # convert left /right joints to original image space (X, Y are aligned to image)
            pred_lhand_joints_bbox = convert_smpl_to_bbox(
                pred_lhand_joints, camScale, camTrans)
            pred_lhand_joints_img = convert_bbox_to_oriIm(
                pred_lhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
            integral_output['pred_lhand_joints_img'] = pred_lhand_joints_img
            #self.lhand = pred_lhand_joints_img

            pred_rhand_joints_bbox = convert_smpl_to_bbox(
                pred_rhand_joints, camScale, camTrans)
            pred_rhand_joints_img = convert_bbox_to_oriIm(
                pred_rhand_joints_bbox, bbox_scale_ratio, bbox_top_left, image_shape[1], image_shape[0])
            integral_output['pred_rhand_joints_img'] = pred_rhand_joints_img
            #self.rhand = pred_rhand_joints_img

            # keep hand info
            r_hand_local_orient_body = body_info['pred_rotmat'][:, 21] # rot-mat
            r_hand_global_orient_body = self.transfer_rotation(
                smplx_model, pred_rotmat,
                torch.from_numpy(r_hand_local_orient_body).cuda(),
                21, 'l2g', 'aa').numpy().reshape(1, 3) # aa
            r_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(r_hand_local_orient_body) # rot-mat -> aa

            l_hand_local_orient_body = body_info['pred_rotmat'][:, 20]
            l_hand_global_orient_body = self.transfer_rotation(
                smplx_model, pred_rotmat,
                torch.from_numpy(l_hand_local_orient_body).cuda(),
                20, 'l2g', 'aa').numpy().reshape(1, 3)
            l_hand_local_orient_body = gu.rotation_matrix_to_angle_axis(l_hand_local_orient_body) # rot-mat -> aa

            r_hand_local_orient_hand = None
            r_hand_global_orient_hand = None
            if right_hand_local_orient is not None:
                r_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                    right_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
                r_hand_global_orient_hand = right_hand_global_orient.detach().cpu().numpy().reshape(1, 3)

            l_hand_local_orient_hand = None
            l_hand_global_orient_hand = None
            if left_hand_local_orient is not None:
                l_hand_local_orient_hand = gu.rotation_matrix_to_angle_axis(
                    left_hand_local_orient).detach().cpu().numpy().reshape(1, 3)
                l_hand_global_orient_hand = left_hand_global_orient.detach().cpu().numpy().reshape(1, 3)

            # poses and rotations related to hands
            integral_output['left_hand_local_orient_body'] = l_hand_local_orient_body
            integral_output['left_hand_global_orient_body'] = l_hand_global_orient_body
            integral_output['right_hand_local_orient_body'] = r_hand_local_orient_body
            integral_output['right_hand_global_orient_body'] = r_hand_global_orient_body

            integral_output['left_hand_local_orient_hand'] = l_hand_local_orient_hand
            integral_output['left_hand_global_orient_hand'] = l_hand_global_orient_hand
            integral_output['right_hand_local_orient_hand'] = r_hand_local_orient_hand
            integral_output['right_hand_global_orient_hand'] = r_hand_global_orient_hand

            integral_output['pred_left_hand_pose'] = left_hand_pose.detach().cpu().numpy()
            integral_output['pred_right_hand_pose'] = right_hand_pose.detach().cpu().numpy()

            # predicted hand betas, cameras, top-left corner and center
            left_hand_betas = None
            left_hand_camera = None
            left_hand_bbox_scale = None
            left_hand_bbox_top_left = None
            if hand_info is not None and hand_info['left_hand'] is not None:
                left_hand_betas = hand_info['left_hand']['pred_hand_betas']
                left_hand_camera = hand_info['left_hand']['pred_camera']
                left_hand_bbox_scale = hand_info['left_hand']['bbox_scale_ratio']
                left_hand_bbox_top_left = hand_info['left_hand']['bbox_top_left']

            right_hand_betas = None
            right_hand_camera = None
            right_hand_bbox_scale = None
            right_hand_bbox_top_left = None
            if hand_info is not None and hand_info['right_hand'] is not None:
                right_hand_betas = hand_info['right_hand']['pred_hand_betas']
                right_hand_camera = hand_info['right_hand']['pred_camera']
                right_hand_bbox_scale = hand_info['right_hand']['bbox_scale_ratio']
                right_hand_bbox_top_left = hand_info['right_hand']['bbox_top_left']

            integral_output['pred_left_hand_betas'] = left_hand_betas
            integral_output['left_hand_camera'] = left_hand_camera
            integral_output['left_hand_bbox_scale_ratio'] = left_hand_bbox_scale
            integral_output['left_hand_bbox_top_left'] = left_hand_bbox_top_left

            integral_output['pred_right_hand_betas'] = right_hand_betas
            integral_output['right_hand_camera'] = right_hand_camera
            integral_output['right_hand_bbox_scale_ratio'] = right_hand_bbox_scale
            integral_output['right_hand_bbox_top_left'] = right_hand_bbox_top_left

            integral_output_list.append(integral_output)



        return integral_output_list

def main(args=None):
	rclpy.init(args=args)
	smpl_dir = '/home/student/frankmocap/extra_data/smpl'
	body_regressor_checkpoint = '/home/student/frankmocap/extra_data/body_module/pretrained_weights/2020_05_31-00_50_43-best-51.749683916568756.pt'
	hand_regressor_checkpoint = '/home/student/frankmocap/extra_data/hand_module/pretrained_weights/pose_shape_best.pth'
	    # Set Visualizer
	
	Mocap_ergonomic = BodyMocap(body_regressor_checkpoint, hand_regressor_checkpoint, smpl_dir)
	rclpy.spin(Mocap_ergonomic)  #loop

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	Mocap_ergonomic.destroy_node()
	rclpy.shutdown()
