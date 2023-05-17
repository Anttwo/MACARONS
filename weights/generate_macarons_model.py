import sys
import os
import argparse
sys.path.append(os.path.abspath('../'))
from macarons.networks.Macarons import *

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='Script to generate a full macarons model ready for training.')
    parser.add_argument('-p', '--pretrained', action='store_true',
                        help='Decides which weights should be used as default weights when initialized the macarons '
                             'model. '
                             'If option is used, pretrained weights will be used as default weights for both occupancy '
                             'and coverage gain modules. '
                             'Else, simple initialized weights will be used as default weights. '
                             'Default weights are not used if the following options -o or -c are provided.')
    parser.add_argument('-o', '--occupancy', type=str, help='Weights file to use when initializing the occupancy '
                                                            'module of the macarons model.')
    parser.add_argument('-c', '--coverage_gain', type=str, help='Weights file to use when initializing the coverage '
                                                                'gain module of the macarons model.')

    args = parser.parse_args()

    if args.pretrained:
        pretrained = True
        scone_occ_model_name = "pretrained_scone_occ.pth"
        scone_vis_model_name = "pretrained_scone_vis.pth"
    else:
        pretrained = False
        scone_occ_model_name = "initialized_scone_occ.pth"
        scone_vis_model_name = "initialized_scone_vis.pth"

    if args.occupancy:
        scone_occ_model_name = args.occupancy
    if args.coverage_gain:
        scone_vis_model_name = args.coverage_gain

    # Useful to load weights from a trained depth model
    pretrained_depth_weights_path = None
    # pretrained_depth_weights_path = "epoch_50_jz_model_macarons_memory_self_supervised_alpha_2_regularity_0.001_ssim_5_1.5_seed_8_9_warmup_200_schedule_lr_0.0001n_freeze_50_capacity_1000_new_lr_dist_th_17.0_zeros_warp_git_loss.pth"
    # pretrained_depth_weights_path = os.path.join(macarons_weights_path, 'trained_macarons.pth')
    load_depth_weights_from_trained_macarons = True

    load_scone_modules = True
    pretrained_occupancy_model_path = os.path.join(scone_weights_path,
                                                   os.path.join("occupancy/", scone_occ_model_name))
    pretrained_visibility_model_path = os.path.join(scone_weights_path,
                                                    os.path.join("coverage_gain/", scone_vis_model_name))

    # if load_pretrained_scone_modules:
    #     pretrained_occupancy_model_path = os.path.join(scone_weights_path,
    #                                                    "occupancy/best_unval_jz_model_scone_occ_mse_warmup_1000_schedule_lr_0.0001.pth")
    #     # pretrained_visibility_model_path = "best_unval_jz_model_scone_vis_surface_coverage_gain_constant_epsilon_kl_divergence_warmup_1000_schedule_lr_0.0001.pth"
    #     # pretrained_visibility_model_path = "best_unval_jz_model_scone_vis_surface_coverage_gain_constant_epsilon_uncentered_l1_warmup_1000_schedule_lr_0.0001_sigmoid.pth"
    #     pretrained_visibility_model_path = os.path.join(scone_weights_path,
    #                                                     "coverage_gain/best_unval_jz_model_scone_vis_surface_coverage_gain_constant_epsilon_uncentered_l1_warmup_1000_schedule_lr_0.0001_sigmoid_tmcs.pth")
    #     # pretrained_visibility_model_path = "best_unval_jz_model_scone_vis_surface_coverage_gain_constant_epsilon_kl_divergence_warmup_1000_schedule_lr_0.0001.pth"
    # else load_initialized_scone_modules:
    #     pretrained_occupancy_model_path = os.path.join(scone_weights_path,
    #                                                    "occupancy/unvalidated_jz_model_scone_occ_mse_warmup_1000_schedule_lr_0.0001_init_single.pth")
    #     pretrained_visibility_model_path = os.path.join(scone_weights_path,
    #                                                     "coverage_gain/unvalidated_jz_model_scone_vis_surface_coverage_gain_constant_epsilon_uncentered_l1_warmup_1000_schedule_lr_0.0001_sigmoid_tmcs_init_single.pth")

    # Set our device:
    numGPU = 0
    if torch.cuda.is_available():
        device = torch.device("cuda:" + str(numGPU))
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    print(device)

    # print("\nLoading depth model...")
    # # depth_model = load_pretrained_module_for_macarons(pretrained_depth_model_path, device)
    # depth_model = create_many_depth_model(learn_pose=False, device=device)
    #
    # print("\nLoading occupancy model...")
    # occupancy_model = load_pretrained_module_weights_for_macarons(SconeOcc(), pretrained_occupancy_model_path, True,
    #                                                               device)
    #
    # print("\nLoading visibility model...")
    # visibility_model = load_pretrained_module_weights_for_macarons(SconeVis(), pretrained_visibility_model_path, True,
    #                                                                device)
    #
    # model = Macarons(depth_model=depth_model,
    #                  occupancy_model=occupancy_model,
    #                  visibility_model=visibility_model).to(device)

    model = create_macarons_model(learn_pose=False, device=device)
    if pretrained_depth_weights_path is not None:
        print("Also loading pretrained depth weights...")
        if load_depth_weights_from_trained_macarons:
            model = load_pretrained_depth_weights_from_macarons(model, pretrained_depth_weights_path,
                                                                True, device)
        else:
            model.depth.depth = load_pretrained_module_weights_for_macarons(model.depth.depth,
                                                                            pretrained_depth_weights_path,
                                                                            True, device)

    if load_scone_modules:
        model.scone.occupancy = load_pretrained_module_weights_for_macarons(model.scone.occupancy,
                                                                            pretrained_occupancy_model_path,
                                                                            True, device)
        model.scone.visibility = load_pretrained_module_weights_for_macarons(model.scone.visibility,
                                                                             pretrained_visibility_model_path,
                                                                             True, device)

        print("Loaded scone modules.")
        print("Scone occ module:", pretrained_occupancy_model_path)
        print("Scone vis module:", pretrained_visibility_model_path)

    else:
        scone_occ = SconeOcc().to(device)
        scone_vis = SconeVis(use_sigmoid=True).to(device)

        print("\nInitialize SCONE occupancy module...")
        torch.manual_seed(9)
        if pretrained_occupancy_model_path is not None:
            pretrained_scone_occ = load_pretrained_module_weights_for_macarons(model.scone.occupancy,
                                                                               pretrained_occupancy_model_path,
                                                                               True, device)
        else:
            pretrained_scone_occ = None
        initialize_scone_occ_weights(scone_occ, from_previous_model=pretrained_scone_occ)

        print("\nInitialize SCONE coverage gain module...")
        torch.manual_seed(9)
        if pretrained_visibility_model_path is not None:
            pretrained_scone_vis = load_pretrained_module_weights_for_macarons(model.scone.visibility,
                                                                               pretrained_visibility_model_path,
                                                                               True, device)
        else:
            pretrained_scone_vis = None
        initialize_scone_vis_weights(scone_vis, from_previous_model=pretrained_scone_vis)

        model.scone.occupancy = scone_occ
        model.scone.visibility = scone_vis
        print("Created vanilla scone modules.")

    # Save full initialized model
    if load_scone_modules:
        if pretrained:
            model_path = 'pretrained_macarons'
        else:
            model_path = 'initialized_macarons'
    else:
        model_path = 'scratch_macarons'

    if pretrained_depth_weights_path is not None:
        model_path += "_with_depth"
        if load_depth_weights_from_trained_macarons:
            model_path += "_from_trained_macarons"

    # model_path += "_debug"  # todo: To remove
    model_path += ".pth"

    model_path = os.path.join(macarons_weights_path, model_path)
    print("\nModel path:", model_path)
    torch.save(model.state_dict(), model_path)
    print("Full model saved.\n")

    print("\nNumber of parameters in depth module:",
          sum(p.numel() for p in model.depth.depth.parameters() if p.requires_grad)/1e6)

    print("\nNumber of parameters in occupancy module:",
          sum(p.numel() for p in model.scone.occupancy.parameters() if p.requires_grad)/1e6)

    print("\nNumber of parameters in visibility module:",
          sum(p.numel() for p in model.scone.visibility.parameters() if p.requires_grad)/1e6)

    print("\nTotal number of parameters:",
          (sum(p.numel() for p in model.scone.parameters() if p.requires_grad)
           + sum(p.numel() for p in model.depth.parameters() if p.requires_grad))/1e6)
