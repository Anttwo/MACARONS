import sys
import os
import json
import time
import argparse

sys.path.append(os.path.abspath('../../'))

from macarons.utility.macarons_utils import *

dir_path = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(dir_path, "./")


def get_cullback_renderer(image_height, image_width, ambient_light_intensity, cameras, device,
                          max_faces_per_bin):
    raster_settings = RasterizationSettings(
        image_size=(image_height, image_width),
        # max_faces_per_bin=500000,
        max_faces_per_bin=200000,
        # bin_size=50,
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=True
    )

    lights = AmbientLights(ambient_color=((ambient_light_intensity,
                                           ambient_light_intensity,
                                           ambient_light_intensity),),
                           device=device)

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    cullback_renderer = MeshRendererWithFragments(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return cullback_renderer


def main(params):
    # Setup the device
    device = setup_device(params)

    # Create dataloader
    _, _, dataloader = get_dataloader(train_scenes=[],
                                      val_scenes=[],
                                      test_scenes=params.scenes,
                                      batch_size=1,
                                      ddp=False, jz=False,
                                      world_size=None, ddp_rank=None,
                                      data_path=params.data_path,
                                      use_occupied_pose=False)

    # Default camera to initialize renderers
    n_camera = 1
    camera_dist = [10 * params.scene_scale_factor] * n_camera  # 10
    camera_elev = [30] * n_camera
    camera_azim = [260] * n_camera  # 160
    R, T = look_at_view_transform(camera_dist, camera_elev, camera_azim)
    zfar = params.zfar
    fov_camera = FoVPerspectiveCameras(R=R, T=T, zfar=zfar, device=device)

    renderer = get_rgb_renderer(image_height=params.image_height,
                                image_width=params.image_width,
                                ambient_light_intensity=params.ambient_light_intensity,
                                cameras=fov_camera,
                                device=device,
                                max_faces_per_bin=200000
                                )

    cullback_renderer = get_cullback_renderer(image_height=params.image_height,
                                              image_width=params.image_width,
                                              ambient_light_intensity=params.ambient_light_intensity,
                                              cameras=fov_camera,
                                              device=device,
                                              max_faces_per_bin=200000)

    for batch in range(len(dataloader.dataset)):
        scene_dict = dataloader.dataset[batch]

        scene_names = [scene_dict['scene_name']]
        obj_names = [scene_dict['obj_name']]
        all_settings = [scene_dict['settings']]

        batch_size = len(scene_names)

        for i_scene in range(batch_size):
            mesh = None
            torch.cuda.empty_cache()

            scene_name = scene_names[i_scene]
            obj_name = obj_names[i_scene]
            settings = all_settings[i_scene]
            settings = Settings(settings, device, params.scene_scale_factor)

            print("\nScene name:", scene_name)
            print("-------------------------------------")
            scene_path = os.path.join(dataloader.dataset.data_path, scene_name)
            mesh_path = os.path.join(scene_path, obj_name)

            # Load mesh
            mesh = load_scene(mesh_path, params.scene_scale_factor, device,
                              mirror=False, mirrored_axis=None)

            print("Vertices shape:", mesh.verts_list()[0].shape)
            print("Min:", torch.min(mesh.verts_list()[0], dim=0)[0],
                  "\nMax:", torch.max(mesh.verts_list()[0], dim=0)[0])

            # Initialize camera
            camera = Camera(x_min=settings.camera.x_min, x_max=settings.camera.x_max,
                            pose_l=settings.camera.pose_l, pose_w=settings.camera.pose_w, pose_h=settings.camera.pose_h,
                            pose_n_elev=settings.camera.pose_n_elev, pose_n_azim=settings.camera.pose_n_azim,
                            n_interpolation_steps=1, zfar=params.zfar,
                            renderer=renderer,
                            device=device,
                            contrast_factor=settings.camera.contrast_factor,
                            gathering_factor=params.gathering_factor,
                            occupied_pose_data=None,
                            save_dir_path=None,
                            mirrored_scene=False,
                            mirrored_axis=None)  # Change or remove this path during inference or test

            start_position = settings.camera.start_positions[0]
            camera.initialize_camera(start_cam_idx=start_position)

            print("Camera:")
            print("x_min, x_max:", settings.camera.x_min, settings.camera.x_max)
            print("pose_l, pose_w, pose_h:", settings.camera.pose_l, settings.camera.pose_w, settings.camera.pose_h)

            # We compute 3D coordinates of all camera poses
            all_X_idx = torch.zeros(0, 3, device=device).long()
            for pose_key in camera.pose_space.keys():
                pose_idx = torch.Tensor([int(c) for c in pose_key[1:-1].split(',')[:3]]).long().to(device)
                all_X_idx = torch.vstack((all_X_idx, pose_idx))

            all_X_idx = torch.unique(all_X_idx, dim=0).long()
            print("Indices computed.")

            occupied_pose = torch.zeros(len(all_X_idx), device=device).bool()

            for i in range(len(all_X_idx)):
                pose_idx = all_X_idx[i]
                full_idx = torch.cat((pose_idx, torch.zeros(2, device=device).long()), dim=0)
                pose_key = camera.get_key_from_idx(full_idx)
                pose = camera.pose_space[pose_key]
                X_cam, V_cam, fov_camera = camera.get_camera_parameters_from_pose(pose)

                empty_fov = camera.is_fov_empty(mesh=mesh, fov_camera=fov_camera)

                while empty_fov:
                    print("Empty fov. Try another V_cam...")
                    elev = -90. + 180. * (1 + np.random.randint(low=0, high=camera.pose_n_elev)) / (
                                camera.pose_n_elev + 1)
                    azim = 360 * np.random.randint(low=0, high=camera.pose_n_azim) / camera.pose_n_azim
                    V_cam[..., 0] = elev
                    V_cam[..., 1] = azim
                    fov_camera = camera.get_fov_camera_from_XV(X_cam, V_cam)
                    empty_fov = camera.is_fov_empty(mesh=mesh, fov_camera=fov_camera)

                camera.fov_camera = fov_camera
                camera.renderer = renderer
                images, _ = camera.capture_image(mesh=mesh, fov_camera=fov_camera, save_frame=False)

                camera.renderer = cullback_renderer
                back_images, _ = camera.capture_image(mesh=mesh, fov_camera=fov_camera, save_frame=False)

                diff_images = (images - back_images)[..., :3]
                diff_images = diff_images.abs().mean(-1, keepdim=True)
                occupied_pose[i] = diff_images.mean().item() > 1e-3

            print("Occupied poses computed:", occupied_pose.sum())
            occupied_pose_path = os.path.join(scene_path, 'occupied_pose.pt')
            occupied_pose_dict = {}
            occupied_pose_dict['X_idx'] = all_X_idx
            occupied_pose_dict['occupied'] = occupied_pose

            torch.save(occupied_pose_dict, occupied_pose_path)
            print("Occupied poses saved to:", occupied_pose_path)


if __name__ == '__main__':
    t0 = time.time()

    # Parser
    parser = argparse.ArgumentParser(description='Script to generate additional ground truth data from large 3D scene '
                                                 'meshes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "generate_scene_data_config.json".')

    args = parser.parse_args()

    if args.config:
        json_name = args.config
    else:
        json_name = "generate_scene_data_config.json"

    # Load config file and generate data
    print("Using config file:")
    print(json_name)

    json_path = os.path.join(configs_dir, json_name)
    params = load_params(json_path)
    main(params)
    print("Generation finished after", time.time() - t0, "seconds.")
