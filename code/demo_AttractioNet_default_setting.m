% demo_AttractioNet demonstrates how to use AttractioNet for extracting the
% bounding box proposals from ilsvrc 2015/2014, the standard officical version.

%clc;
clear;
close all;
run('startup');
caffe.reset_all();
%************************* SET GPU/CPU DEVICE *****************************
% 1-indexed
gpu_id = 1;
total_chunk = 4;
curr_chunk = 1;
which_set = 'val'; %'train';
result_name = 'hyli_default_settting';
top_k = 1000;

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
% box_prop_conf.num_iterations = 1;
% box_prop_conf.nms_iou_thrs = 0.5;
% box_prop_conf.max_per_image = 2000;

fprintf('AttractioNet configuration params:\n');
disp(box_prop_conf);
model.scales   = box_prop_conf.scales;
model.max_size = box_prop_conf.max_size;
fprintf('AttractioNet Model:\n');
disp(model);

%**************************************************************************
%****************************** READ IMAGE ********************************
% dataset
save_path = ['./box_proposals/author_provide/' result_name];
mkdir_if_missing(save_path);

ld = load('../cvpr17_proposal/data/imdb/train_val_list.mat');
if strcmp(which_set, 'train');
    full_im_list = ld.train_list;
elseif strcmp(which_set, 'val');
    full_im_list = ld.val_list;
end
full_num_images = length(full_im_list);
ck_interval = ceil(full_num_images/total_chunk);
start_ind = 1 + (curr_chunk-1)*ck_interval;
end_ind = min(ck_interval + (curr_chunk-1)*ck_interval, full_num_images);
num_images = end_ind - start_ind + 1;
im_list = full_im_list(start_ind : end_ind);  % part of the whole set

root_folder{1} = '../cvpr17_proposal/data/datasets/ilsvrc14_det/ILSVRC2013_DET_val';
root_folder{2} = '../cvpr17_proposal/data/datasets/ilsvrc14_det/ILSVRC2014_DET_train';

%**************************************************************************
%*************************** RUN AttractioNet *****************************
box_result(num_images).name = '';
box_result(num_images).box = [];
save_mat_file = [save_path sprintf('/%s_ck%d_absInd_%d_%d_total%d.mat', ...
    which_set, curr_chunk, start_ind, end_ind, full_num_images)];

t = tic;
for i = 1 : num_images
   
    name_temp = im_list{i}(21:end);
    box_result(i).name = name_temp;   
    if name_temp(1) == 't'
        im_path = fullfile(root_folder{2}, name_temp(7:end));
    elseif name_temp(1) == 'v'
        im_path = fullfile(root_folder{1}, name_temp(5:end));
    end
       
    try
        image = imread(im_path);
    catch lasterror
        % hah, annoying data issues
        if strcmp(lasterror.identifier, 'MATLAB:imagesci:jpg:cmykColorSpace')
            warning('converting %s from CMYK to RGB', im_path);
            cmd = ['convert ' im_path ' -colorspace CMYK -colorspace RGB ' im_path];
            system(cmd);
            image = imread(im_path);
        else
            error(lasterror.message);
        end
    end
    boxes = AttractioNet_original(model, image, box_prop_conf);
    box_result(i).box = single(boxes(1:min(top_k, size(boxes,1)), 1:4));
    
    if mod(i, 50) == 1 || i == num_images
        take = toc(t)/(3600*50);
        time_left = take*(num_images-i);
        fprintf('%s, ck# %d (%d-%d), gpu# %d, progress i/total, %d/%d, %.3f hrs left ...\n', ...
            which_set, curr_chunk, start_ind, end_ind, gpu_id, i, num_images, time_left);
        t = tic;
    end
end
save(save_mat_file, 'box_result', '-v7.3');
caffe.reset_all();

