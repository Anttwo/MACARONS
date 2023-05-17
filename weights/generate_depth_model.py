import sys
import os
sys.path.append(os.path.abspath('../'))
from macarons.networks.ManyDepth import *

if __name__ == "__main__":
    # Set our device:
    numGPU = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(numGPU))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(device)

    # Save and load pretrained feature extractor
    fe_path = 'feature_extractor.pth'
    fe_path = os.path.join(weights_path, fe_path)
    save_feature_extractor_from_resnet18(fe_path, device)
    print("Feature extractor saved.\n")
    feature_extractor = load_feature_extractor(fe_path, device)

    # Save and load pretrained depth decoder
    decoder_path = 'depth_decoder.pth'
    decoder_path = os.path.join(weights_path, decoder_path)
    save_depth_decoder_from_resnet18(decoder_path,
                                     device=device,
                                     feature_extractor=feature_extractor)
    print("Depth decoder saved.\n")
    depth_decoder = load_depth_decoder(decoder_path, device=device)
    print("Use input image skip connection in depth model:", depth_decoder.use_input_image_in_skip_connection, '\n')

    # Save and load pretrained pose decoder
    pose_decoder_path = 'pose_decoder.pth'
    pose_decoder_path = os.path.join(weights_path, pose_decoder_path)
    save_pose_decoder_from_resnet18(pose_decoder_path,
                                    device=device)
    pose_decoder = load_pose_decoder(pose_decoder_path, device=device)
    print("Pose decoder saved.\n")

    # Save full pretrained model
    model = ManyDepth(depth_decoder=depth_decoder, pose_decoder=pose_decoder)
    model_path = 'pretrained_many_depth'
    if model.learn_pose:
        model_path += "_pose"
    if n_alpha > 1:
        model_path += "_alpha_" + str(n_alpha)
    model_path += ".pth"
    model_path = "depth_with_resnet_imagenet_weights.pth"
    # model_path = "depth_with_resnet_imagenet_weights_debug.pth"  # todo: To remove
    model_path = os.path.join(weights_path, model_path)
    torch.save(model, model_path)
    print("Full model saved.\n")

    print("Model path:", model_path)
    print("Number of parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
    print("input_height:", model.input_height)
    print("input_width:", model.input_width)
    print("n_alpha:", model.depth_decoder.cost_volume_builder.n_alpha)
    print("d_min:", model.d_min)
    print("d_max:", model.d_max)
    print("n_depth:", model.n_depth)
    print("pose_factor:", model.pose_factor)
    print("learn_pose:", model.learn_pose)
