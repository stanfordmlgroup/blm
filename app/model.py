import cv2
import torch
import torch.nn as nn
import numpy as np
import logging

from utils_model import compute_boxes_and_sizes, get_upsample_output, get_box_and_dot_maps, get_boxed_img


class LSCCNN(nn.Module):
    def __init__(self, name='scale_4', checkpoint_path=None, output_downscale=2,
                 PRED_DOWNSCALE_FACTORS=(8, 4, 2, 1), GAMMA=(1, 1, 2, 4), NUM_BOXES_PER_SCALE=3):

        super(LSCCNN, self).__init__()
        self.name = name
        if torch.cuda.is_available():
            self.rgb_means = torch.cuda.FloatTensor([104.008, 116.669, 122.675])
        else:
            self.rgb_means = torch.FloatTensor([104.008, 116.669, 122.675])
        self.rgb_means = torch.autograd.Variable(self.rgb_means, requires_grad=False).unsqueeze(0).unsqueeze(
            2).unsqueeze(3)

        self.BOXES, self.BOX_SIZE_BINS = compute_boxes_and_sizes(PRED_DOWNSCALE_FACTORS, GAMMA, NUM_BOXES_PER_SCALE)
        self.output_downscale = output_downscale

        in_channels = 3
        self.relu = nn.ReLU(inplace=True)
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.convA_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convA_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convA_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convA_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convA_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convB_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convB_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convB_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convB_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convB_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convC_1 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.convC_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convC_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convC_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convC_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.convD_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.convD_2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convD_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convD_4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.convD_5 = nn.Conv2d(32, 4, kernel_size=3, padding=1)

        self.conv_before_transpose_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.transpose_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_1_1 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.transpose_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_3 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_3_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.transpose_4_1_a = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.transpose_4_1_b = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_1 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_2 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=4, padding=0, output_padding=1)
        self.conv_after_transpose_4_2 = nn.Conv2d(256, 64, kernel_size=3, padding=1)

        self.transpose_4_3 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_after_transpose_4_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.conv_middle_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv_middle_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_middle_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv_mid_4 = nn.Conv2d(512, 256, kernel_size=3, padding=1)

        self.conv_lowest_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_lowest_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_lowest_4 = nn.Conv2d(256, 128, kernel_size=3, padding=1)

        self.conv_scale1_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv_scale1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_scale1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        if checkpoint_path is not None:
            if torch.cuda.is_available():
                self.load_state_dict(torch.load(checkpoint_path))
            else:
                # https://towardsdatascience.com/how-to-save-and-load-a-model-in-pytorch-with-a-complete-example-c2920e617dee
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.load_state_dict(checkpoint['state_dict'])

    def forward(self, x):
        mean_sub_input = x
        mean_sub_input -= self.rgb_means

        logging.info("Starting forward pass ...")

        #################### Stage 1 ##########################

        logging.info("Stage 1")
        
        main_out_block1 = self.relu(self.conv1_2(self.relu(self.conv1_1(mean_sub_input))))
        main_out_pool1 = self.pool1(main_out_block1)

        main_out_block2 = self.relu(self.conv2_2(self.relu(self.conv2_1(main_out_pool1))))
        main_out_pool2 = self.pool2(main_out_block2)

        main_out_block3 = self.relu(self.conv3_3(self.relu(self.conv3_2(self.relu(self.conv3_1(main_out_pool2))))))
        main_out_pool3 = self.pool3(main_out_block3)

        main_out_block4 = self.relu(self.conv4_3(self.relu(self.conv4_2(self.relu(self.conv4_1(main_out_pool3))))))
        main_out_pool4 = self.pool3(main_out_block4)

        main_out_block5 = self.relu(self.conv_before_transpose_1(
            self.relu(self.conv5_3(self.relu(self.conv5_2(self.relu(self.conv5_1(main_out_pool4))))))))

        main_out_rest = self.convA_5(self.relu(
            self.convA_4(self.relu(self.convA_3(self.relu(self.convA_2(self.relu(self.convA_1(main_out_block5)))))))))
        if self.name == "scale_1":
            return main_out_rest
        
        ################## Stage 2 ############################

        logging.info("Stage 2")

        sub1_out_conv1 = self.relu(self.conv_mid_4(self.relu(
            self.conv_middle_3(self.relu(self.conv_middle_2(self.relu(self.conv_middle_1(main_out_pool3))))))))
        sub1_transpose = self.relu(self.transpose_1(main_out_block5))
        sub1_after_transpose_1 = self.relu(self.conv_after_transpose_1_1(sub1_transpose))

        sub1_concat = torch.cat((sub1_out_conv1, sub1_after_transpose_1), dim=1)

        sub1_out_rest = self.convB_5(self.relu(
            self.convB_4(self.relu(self.convB_3(self.relu(self.convB_2(self.relu(self.convB_1(sub1_concat)))))))))
        if self.name == "scale_2":
            return main_out_rest, sub1_out_rest
        ################# Stage 3 ############################

        logging.info("Stage 3")

        sub2_out_conv1 = self.relu(self.conv_lowest_4(self.relu(
            self.conv_lowest_3(self.relu(self.conv_lowest_2(self.relu(self.conv_lowest_1(main_out_pool2))))))))
        sub2_transpose = self.relu(self.transpose_2(sub1_out_conv1))
        sub2_after_transpose_1 = self.relu(self.conv_after_transpose_2_1(sub2_transpose))

        sub3_transpose = self.relu(self.transpose_3(main_out_block5))
        sub3_after_transpose_1 = self.relu(self.conv_after_transpose_3_1(sub3_transpose))

        sub2_concat = torch.cat((sub2_out_conv1, sub2_after_transpose_1, sub3_after_transpose_1), dim=1)

        sub2_out_rest = self.convC_5(self.relu(
            self.convC_4(self.relu(self.convC_3(self.relu(self.convC_2(self.relu(self.convC_1(sub2_concat)))))))))

        if self.name == "scale_3":
            return main_out_rest, sub1_out_rest, sub2_out_rest
        ################# Stage 4 ############################

        logging.info("Stage 4")

        sub4_out_conv1 = self.relu(
            self.conv_scale1_3(self.relu(self.conv_scale1_2(self.relu(self.conv_scale1_1(main_out_pool1))))))

        # TDF 1
        tdf_4_1_a = self.relu(self.transpose_4_1_a(main_out_block5))
        tdf_4_1_b = self.relu(self.transpose_4_1_b(tdf_4_1_a))
        after_tdf_4_1 = self.relu(self.conv_after_transpose_4_1(tdf_4_1_b))

        # TDF 2
        tdf_4_2 = self.relu(self.transpose_4_2(sub1_out_conv1))
        after_tdf_4_2 = self.relu(self.conv_after_transpose_4_2(tdf_4_2))

        # TDF 3
        tdf_4_3 = self.relu(self.transpose_4_3(sub2_out_conv1))
        after_tdf_4_3 = self.relu(self.conv_after_transpose_4_3(tdf_4_3))

        sub4_concat = torch.cat((sub4_out_conv1, after_tdf_4_1, after_tdf_4_2, after_tdf_4_3), dim=1)
        sub4_out_rest = self.convD_5(self.relu(
            self.convD_4(self.relu(self.convD_3(self.relu(self.convD_2(self.relu(self.convD_1(sub4_concat)))))))))

        logging.info("Forward pass completed")
        if self.name == "scale_4":
            return main_out_rest, sub1_out_rest, sub2_out_rest, sub4_out_rest


    def handle_image_size(self, image):
        # width_new = width_orig = image.shape[1]
        # height_new = height_orig = image.shape[0]
        h, w, _ = image.shape

        if w % 16 == 0 and h % 16 == 0:
            return image

        pad_w = 0 if w % 16 == 0 else 16 - (w % 16)
        pad_h = 0 if h % 16 == 0 else 16 - (h % 16)

        logging.info("Padding image for processing (+{}, +{}). Resulting size: ({}, {})".format(pad_w, pad_h, w + pad_w, h + pad_h))
        image_padded = cv2.copyMakeBorder (image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT)

        return image_padded, w, h
    

    def predict_single_image(self, image, emoji, nms_thresh=0.25, thickness=2, multi_colours=True):
        # resize(pad) image for processing if necessary
        img_in, w, h = self.handle_image_size(image)

        img_tensor = torch.from_numpy(img_in.transpose((2, 0, 1)).astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            out = self.forward(img_tensor)
        
        out = get_upsample_output(out, self.output_downscale)
        pred_dot_map, pred_box_map = get_box_and_dot_maps(out, nms_thresh, self.BOXES)
        img_emoji = get_boxed_img(img_in, emoji, pred_box_map, pred_box_map, pred_dot_map, self.output_downscale,
                                self.BOXES, self.BOX_SIZE_BINS, thickness=thickness, multi_colours=multi_colours)

        # crop image to original size
        img_out = img_emoji[0:h, 0:w]
        
        return pred_dot_map, pred_box_map, img_out