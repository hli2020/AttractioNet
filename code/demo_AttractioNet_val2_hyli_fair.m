% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2015/2014, the standard officical version.

%clc;
clear;
close all;
run('startup');
caffe.reset_all();
%************************* SET GPU/CPU DEVICE *****************************
% By setting gpu_id = 1, the first GPU (one-based counting) is used for
% running the AttractioNet model. By setting gpu_id = 0, then the CPU will
% be used for running the AttractioNet model. Note that I never tested it
% my self the CPU.
gpu_id = 0;
caffe_set_device( gpu_id );
caffe.set_mode_gpu();
%**************************************************************************
%***************************** LOAD MODEL *********************************
model_dir_name = 'AttractioNet_Model';
full_model_dir = fullfile(pwd, 'models-exps', model_dir_name);
assert(exist(full_model_dir,'dir')>0,sprintf('The %s model directory does not exist',full_model_dir));
mat_file_name  = 'box_proposal_model.mat';
model = AttractioNet_load_model(full_model_dir, mat_file_name);
%**************************************************************************
%********************** CONFIGURATION PARAMETERS **************************
disp(' ');
box_prop_conf = AttractioNet_get_defaul_conf();
box_prop_conf.num_iterations = 1;
box_prop_conf.nms_iou_thrs = 0.5;
box_prop_conf.max_per_image = 2000;

fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;
model.max_size = box_prop_conf.max_size;
fprintf('AttractioNet Model:\n');
disp(model);

%**************************************************************************
%****************************** READ IMAGE ********************************
%image_path = fullfile(pwd,'examples','COCO_val2014_000000109798.jpg');
%image_path = fullfile(pwd,'examples','000029.jpg');
%image = imread(image_path);

% dataset
result_name = 'oct_19_fair';
result_path = './box_proposals/author_provide/val2';

root_folder = '/home/hongyang/dataset/imagenet_det/ILSVRC2014_devkit';
fid = fopen([root_folder '/data/det_lists/val2.txt'], 'r');
temp = textscan(fid, '%s%s');
im_list = temp{1}; clear temp;
%gt_path = [root_folder '/../ILSVRC2013_DET_bbox_val/'];
im_path = [root_folder '/../ILSVRC2013_DET_val'];

mkdir_if_missing([result_path '/' result_name]);

%**************************************************************************
%*************************** RUN AttractioNet *****************************
start_im = 1;
end_im = 500; %length(im_list)

for i = start_im : end_im 
    
    res_path = [result_path '/' result_name '/' im_list{i} '.mat'];
    if ~exist(res_path, 'file')
        image = imread([im_path '/' im_list{i} '.JPEG']);
        boxes = AttractioNet_original(model, image, box_prop_conf);
        save(res_path, 'boxes');
    end
    if mod(i, 50) == 1 || i == end_im
        fprintf('i = %d (%d - %d)\n', i, start_im, end_im);
    end
end

caffe.reset_all();

