# Copyright (c) Facebook, Inc. and its affiliates.

import os
import sys
import os.path as osp
import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2

from demo.demo_options import DemoOptions
from bodymocap.body_mocap_api import BodyMocap
from bodymocap.body_bbox_detector import BodyPoseEstimator
import mocap_utils.demo_utils as demo_utils
import mocap_utils.general_utils as gnu
from mocap_utils.timer import Timer


from mrl_cobot.telekinesis_utils import CobotTelekinesis
from renderer import glViewer

def run_body_mocap(args, body_bbox_detector, body_mocap, visualizer):
    #Setup input data to handle different types of inputs
    input_type, input_data = demo_utils.setup_input(args)
    cobot_utils = CobotTelekinesis()
    #Connect to cobot
    if args.cobot:
        cobot_utils.connect_to_cobot(('localhost', 60001))
    
    cur_frame = args.start_frame
    video_frame = 0
    timer = Timer()
    while True:
        timer.tic()
        # load data
        load_bbox = False

        if input_type =='image_dir':
            if cur_frame < len(input_data):
                image_path = input_data[cur_frame]
                img_original_bgr  = cv2.imread(image_path)
            else:
                img_original_bgr = None

            if args.post_proc_eft:
                assert args.openpose_dir is not None

                #Note: current openpose name should be {raw_image_name}_keypoints.json
                f_name = os.path.basename(image_path)[:-4] + "_keypoints.json"
                openpose_file_path = os.path.join(args.openpose_dir,f_name)
                assert os.path.exists(openpose_file_path)
                print(f"Loading openpose data from: {openpose_file_path}")
                openpose_imgcoord, _ = demo_utils.read_openpose_wHand(openpose_file_path,dataset='coco')      #25, 3       #TODO: this works for single person in the image
                assert openpose_imgcoord is not None

        elif input_type == 'bbox_dir':
            if cur_frame < len(input_data):
                print("Use pre-computed bounding boxes")
                image_path = input_data[cur_frame]['image_path']
                hand_bbox_list = input_data[cur_frame]['hand_bbox_list']
                body_bbox_list = input_data[cur_frame]['body_bbox_list']
                img_original_bgr  = cv2.imread(image_path)
                load_bbox = True
            else:
                img_original_bgr = None

        elif input_type == 'video':      
            _, img_original_bgr = input_data.read()
            img_original_bgr = cv2.resize(img_original_bgr, (526, 526))
            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            if args.out_dir is not None:
                image_path = osp.join(args.out_dir, "frames", f"{cur_frame:05d}.jpg")
                if img_original_bgr is not None:
                    video_frame += 1
                    if args.save_frame:
                        gnu.make_subdir(image_path)
                        cv2.imwrite(image_path, img_original_bgr)

        elif input_type == 'webcam':    
            _, img_original_bgr = input_data.read()

            if video_frame < cur_frame:
                video_frame += 1
                continue
            # save the obtained video frames
            image_path = osp.join(args.out_dir, "frames", f"scene_{cur_frame:05d}.jpg")
            if img_original_bgr is not None:
                video_frame += 1
                if args.save_frame:
                    gnu.make_subdir(image_path)
                    cv2.imwrite(image_path, img_original_bgr)
        else:
            assert False, "Unknown input_type"

        cur_frame +=1
        if img_original_bgr is None or cur_frame > args.end_frame:
            break   
        print("--------------------------------------")

        if load_bbox:
            body_pose_list = None
        else:
            body_pose_list, body_bbox_list = body_bbox_detector.detect_body_pose(
                img_original_bgr)
        hand_bbox_list = [None, ] * len(body_bbox_list)

        # save the obtained body & hand bbox to json file
        if args.save_bbox_output: 
            demo_utils.save_info_to_json(args, image_path, body_bbox_list, hand_bbox_list)
            timer.toc(bPrint=True, title="Save bbox")
            continue

        if len(body_bbox_list) < 1: 
            print(f"No body deteced: {image_path}")
            continue

        #Sort the bbox using bbox size 
        # (to make the order as consistent as possible without tracking)
        bbox_size =  [ (x[2] * x[3]) for x in body_bbox_list]
        idx_big2small = np.argsort(bbox_size)[::-1]
        body_bbox_list = [ body_bbox_list[i] for i in idx_big2small ]
        if args.single_person and len(body_bbox_list)>0:
            body_bbox_list = [body_bbox_list[0], ]       

        # Body Pose Regression
        if args.post_proc_eft:  
            pred_output_list = body_mocap.regress_and_eft(img_original_bgr, body_bbox_list, openpose_imgcoord)     #Apply eft post processing with openpose
        else:
            pred_output_list = body_mocap.regress(img_original_bgr, body_bbox_list)
        assert len(body_bbox_list) == len(pred_output_list)

        res_img = img_original_bgr
        
        # show result in the screen
        joint_indices = [0, 6, 9, 14, 17, 19, 21]
        joint_positions_indices = [39, 31]
        cobot_utils.proc_time()
        joint_rotations = cobot_utils.extract_joint_data(pred_output_list[0]["pred_body_pose"][0], joint_indices, 1)
        #print("DEBUG pred_output_list[0]:", pred_output_list[0].keys())
        joint_positions = cobot_utils.extract_joint_data(pred_output_list[0]["pred_joints_img"], joint_positions_indices, 0)
        cobot_utils.proc_time(1)
        r_shoulder_pos = pred_output_list[0]["pred_joints_img"][33]
        l_shoulder_pos = pred_output_list[0]["pred_joints_img"][34]
        scale_factor = 0.45/np.linalg.norm(l_shoulder_pos - r_shoulder_pos) # 45cm Average shoulder width (Men)

        relative_transform, T_torso, T_wrist = cobot_utils.compute_relative_transformation(joint_positions, joint_rotations, scale_factor)

        torso_position = joint_positions[0]
        torso_orientation = joint_rotations[0]

        T_wrist = T_torso @ relative_transform # For testing Wrist reconstruction
        wrist_position = joint_positions[1]
        wrist_orientation = T_wrist[:3, :3]

        res_img = cobot_utils.draw_axes(res_img, torso_position, torso_orientation, rotation_matrix_flag=True)
        res_img = cobot_utils.draw_axes(res_img, wrist_position, wrist_orientation, rotation_matrix_flag=True)

        if args.cobot:
            cobot_utils.proc_time()
            print("[DEBUG] About to send data to cobot...")
            cobot_utils.send_data(relative_transform)
            print("[DEBUG] Sent data successfully.")
            cobot_utils.proc_time(1)

        if not args.no_display:
            cobot_utils.show_image(res_img)

        # save result image
        # if args.out_dir is not None:
        #     demo_utils.save_res_img(args.out_dir, image_path, res_img)

        # save predictions to pkl
        if args.save_pred_pkl:
            demo_type = 'body'
            demo_utils.save_pred_to_pkl(
                args, demo_type, image_path, body_bbox_list, hand_bbox_list, pred_output_list)
        timer.toc(average = True, bPrint=True,title="Time")
        # print(f"Processed : {image_path}")

    #save images as a video
    if not args.no_video_out and input_type in ['image_dir', 'video', 'webcam']:
        demo_utils.gen_video_out(args.out_dir, args.seq_name)

    if input_type =='webcam' and input_data is not None:
        input_data.release()
    cv2.destroyAllWindows()
    cobot_utils.disconnect_from_cobot()


def main():
    args = DemoOptions().parse()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert torch.cuda.is_available(), "Current version only supports GPU"

    # Set bbox detector
    body_bbox_detector = BodyPoseEstimator()
    args.no_video_out = True
    # Set mocap regressor
    use_smplx = args.use_smplx
    args.single_person = True
    checkpoint_path = args.checkpoint_body_smplx if use_smplx else args.checkpoint_body_smpl
    print("use_smplx", use_smplx)
    body_mocap = BodyMocap(checkpoint_path, args.smpl_dir, device, use_smplx)

    # Set Visualizer
    if args.renderer_type in ['pytorch3d', 'opendr']:
        from renderer.screen_free_visualizer import Visualizer
    else:
        from renderer.visualizer import Visualizer
    visualizer = Visualizer(args.renderer_type)
  
    run_body_mocap(args, body_bbox_detector, body_mocap, visualizer)


if __name__ == '__main__':
    main()