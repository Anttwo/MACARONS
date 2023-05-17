import sys
import os
import errno
import argparse
import time

# from ...macarons.utility.idr_torch import size as idr_torch_size

debug = False

sys.path.append(os.path.abspath('../../'))

from macarons.utility.utils import *
from macarons.utility.CustomGeometry import *
from macarons.utility.scone_utils import (
    load_params,
    setup_device,
    get_shapenet_dataloader,
    get_cameras_on_sphere,
)

dir_path = os.path.abspath(os.path.dirname(__file__))
configs_dir = os.path.join(dir_path, "./")


def main(params):
    # Set up the device
    device = setup_device(params)

    # Create dataloader
    train_dataloader, val_dataloader, test_dataloader = get_shapenet_dataloader(batch_size=params.batch_size,
                                                                                ddp=params.ddp, jz=params.jz,
                                                                                world_size=None, ddp_rank=None,
                                                                                test_number=-1,
                                                                                load_obj=True,
                                                                                data_path=params.data_path,
                                                                                shuffle=False)
    _, _, novel_test_dataloader = get_shapenet_dataloader(batch_size=params.batch_size,
                                                          ddp=params.ddp, jz=params.jz,
                                                          world_size=None, ddp_rank=None,
                                                          test_number=-1,
                                                          test_novel=True,
                                                          load_obj=True,
                                                          data_path=params.data_path,
                                                          shuffle=False)

    dataloaders = [train_dataloader, val_dataloader, test_dataloader, novel_test_dataloader]
    dataloader_names = ["training", "validation", "test", "novel categories"]

    if debug:
        dataloaders = [train_dataloader]
        dataloader_names = ["debug training"]

    # Create orthographic rasterizer for space carving
    ortho_rasterizer = make_rasterizer(None, device, accurate=True,
                                       image_size=params.image_size, camera_dist=params.ortho_camera_dist,
                                       elevation=params.ortho_elevation, azim_angle=params.ortho_azim_angle)
    print("Orthograhic rasterizer created.")

    # Create FoVPerspective rasterizer to compute partial point clouds
    fov_dist = 1.5
    fov_elev = -90 + 180 * torch.rand(1)
    fov_azim = 360 * torch.rand(1)
    fov_R, fov_T = look_at_view_transform(dist=fov_dist, elev=fov_elev, azim=fov_azim, device=device)
    fov_cameras = FoVPerspectiveCameras(device=device, R=fov_R, T=fov_T)
    fov_rasterizer = make_screen_rasterizer(params, params.n_view_max, fov_cameras, device)
    print("FoVPerspective rasterizer created.")

    # Get camera candidates
    X_cam_world, candidate_dist, candidate_elev, candidate_azim = get_cameras_on_sphere(params, device,
                                                                                        pole_cameras=params.pole_cameras
                                                                                        )
    candidate_R, candidate_T = look_at_view_transform(dist=candidate_dist,
                                                      elev=candidate_elev,
                                                      azim=candidate_azim,
                                                      device=device)

    sizes = []
    total_n_mesh = np.sum([len(dataloader.dataset) for dataloader in dataloaders])

    for i in range(len(dataloaders)):
        dataloader = dataloaders[i]
        print("\n=====Generating ground truth for", dataloader_names[i], "data=====")
        print("\n" + str(len(dataloader.dataset)), "meshes in this dataloader.")

        for batch, (mesh_dict) in enumerate(dataloader):
            verts = mesh_dict['verts'][0].to(device)
            faces = mesh_dict['faces'][0].to(device)
            atlas = mesh_dict['atlas'][0].to(device)
            paths = mesh_dict['path'][0]

            # Save path
            parent_dir = os.path.dirname(paths)
            save_directory = os.path.join(parent_dir, "tensors")
            try:
                os.makedirs(save_directory)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            surface_save_path = os.path.join(save_directory, 'surface_points.pt')
            part_pc_save_path = os.path.join(save_directory, 'partial_point_clouds.pt')
            occupancy_save_path = os.path.join(save_directory, 'occupancy_field.pt')

            compute_gt = params.recompute \
                         or (not os.path.exists(surface_save_path)) \
                         or (not os.path.exists(part_pc_save_path)) \
                         or (not os.path.exists(occupancy_save_path))

            if compute_gt:
                # ----------Load input mesh----------
                t0 = time.time()

                # Create the mesh object
                mesh = Meshes(
                    verts=[verts],
                    faces=[faces],
                    textures=TexturesAtlas(atlas=[atlas]), )

                # Create an additional, higher resolution mesh for better ground truth surface computation
                surface_verts = adjust_mesh_diagonally(verts, diag_range=params.surface_resolution)
                surface_mesh = Meshes(verts=[surface_verts],
                                      faces=[faces],
                                      textures=TexturesAtlas(atlas=[atlas]), )

                # ----------1. Compute random GT surface points---------------------------------------------------------
                gt_surface = sample_points_on_mesh_surface(verts, faces, params.n_points_surface)
                computed_surface_epsilon = compute_surface_epsilon(gt_surface, quantile=params.epsilon_quantile)
                if params.surface_epsilon_is_constant:
                    surface_epsilon = params.surface_epsilon
                else:
                    surface_epsilon = computed_surface_epsilon

                # ----------2. Compute partial point clouds captured from candidate cameras and their coverage----------
                part_pc = []
                coverage = []

                for j_cam in range(params.n_camera):
                    candidate_camera = FoVPerspectiveCameras(device=device,
                                                             R=candidate_R[j_cam:j_cam + 1],
                                                             T=candidate_T[j_cam:j_cam + 1])

                    # Compute surface points for candidate camera
                    candidate_fragments = fov_rasterizer(surface_mesh.extend(len(candidate_camera)),
                                                         cameras=candidate_camera)

                    depth = candidate_fragments.zbuf
                    surface = project_depth_back_to_3D(depth, candidate_camera) / params.surface_resolution

                    # Compute partial point cloud to be saved
                    part_pc_j = surface[torch.randperm(len(surface))[:params.part_pc_length]].view(-1, 3)

                    # Compute coverage matrix
                    factor = 2
                    surface_ds = surface[torch.randperm(len(surface))[:factor * params.n_points_surface]]
                    coverage_j = torch.min(torch.cdist(gt_surface.double(), surface_ds.double(), p=2.0), dim=-1)[0]
                    coverage_j = torch.heaviside(surface_epsilon - coverage_j,
                                                 values=torch.zeros_like(coverage_j, device=device))
                    part_pc.append(part_pc_j)
                    coverage.append(coverage_j)

                # ----------3. Compute GT occupancy probabilities for random points-------------------------------------

                # First, we sample random points with a uniform distribution in the bounding box
                n_uniform_samples = int(params.n_max_samples * params.sampling_ratio)
                X_uni = sample_X_in_box(x_range=params.sampling_diagonal_range,
                                        n_sample=n_uniform_samples,
                                        device=device).view(1, n_uniform_samples, 3)

                # We also sample some points around the surface, with a gaussian noise
                n_surface_samples = params.n_max_samples - n_uniform_samples
                X_surf = torch.zeros(1, n_surface_samples, 3, device=device)
                X_surf[0] = gt_surface[torch.randint(low=0,
                                                     high=len(gt_surface),
                                                     size=(n_surface_samples,))] + 0.
                X_surf[0] += params.sampling_noise_std * torch.randn(n_surface_samples, 3, device=device)
                x_range = params.sampling_diagonal_range
                X_surf[0][torch.abs(X_surf[0]) > x_range / 2] /= (2 / x_range) * torch.abs(X_surf)[0][
                    torch.abs(X_surf)[0] > x_range / 2]

                # Concatenates uniform and surface points
                X_world = torch.cat((X_uni, X_surf), dim=1)
                # Shuffles points
                shuffled_indices = torch.randperm(len(X_world[0]))
                X_world[0] = X_world[0][shuffled_indices]

                # Compute orthographic depth maps
                ortho_camera = ortho_rasterizer.cameras
                ortho_depth = ortho_rasterizer(surface_mesh.extend(params.n_ortho_camera)).zbuf

                # Compute zbuf for all samples
                X_zbuf = ortho_camera.get_world_to_view_transform().transform_points(X_world * params.surface_resolution
                                                                                     )[..., 2:] / params.surface_resolution

                # Compute corresponding zbuf in orthographic depth
                ortho_mask = (ortho_depth > -1)[..., 0]
                ortho_depth[~ortho_mask] = 100 * params.sampling_diagonal_range * params.surface_resolution
                ortho_depth = ortho_depth[..., 0].unsqueeze(dim=1)

                X_proj = -1. * ortho_camera.get_full_projection_transform().transform_points(
                    X_world * params.surface_resolution)[..., :2].view(params.n_ortho_camera, -1, 1, 2)
                ortho_zbuf = torch.nn.functional.grid_sample(input=ortho_depth,
                                                             grid=X_proj,
                                                             mode='bilinear',
                                                             padding_mode='border'  # 'reflection', 'zeros'
                                                             ) / params.surface_resolution
                ortho_zbuf = ortho_zbuf.view(params.n_ortho_camera, -1, 1)

                # Compute GT occupancy by carving with all depth maps
                occ = ((X_zbuf - ortho_zbuf) > 0.).float().prod(dim=0)

                X_world = X_world.view(-1, 3)

                empty_mask = occ[..., 0] == 0
                full_mask = occ[..., 0] > 0

                X_world = torch.vstack((X_world[full_mask], X_world[empty_mask]))
                occ = torch.vstack((occ[full_mask], occ[empty_mask]))

                X_indices = torch.randperm(params.n_samples_for_occupancy)
                X_world = X_world[:params.n_samples_for_occupancy][X_indices]
                occ = occ[:params.n_samples_for_occupancy][X_indices]

                X_world = torch.vstack((X_world, gt_surface))
                occ = torch.vstack((occ, torch.ones(len(gt_surface), 1, device=device)))
                proxy_points = torch.cat((X_world, occ), dim=-1)

                # ----------4. Save data--------------------------------------------------------------------------------
                torch.save({'surface_points': gt_surface,
                            'epsilon': computed_surface_epsilon,
                            'quantile': params.epsilon_quantile,
                            'resolution': params.surface_resolution},
                           surface_save_path)

                torch.save({'partial_point_cloud': part_pc,
                            'coverage': coverage},
                           part_pc_save_path)

                torch.save({'occupancy_field': proxy_points},
                           occupancy_save_path)

                size = get_memory_size({'test': part_pc}) \
                       + get_memory_size({'test': coverage}) \
                       + get_memory_size({'test': gt_surface}) \
                       + get_memory_size({'test': proxy_points})
                sizes.append(size / 10 ** 6)

                if batch % 20 == 0:
                    print("\n-----Mesh " + str(batch+1) + "/" + str(total_n_mesh) + " done.-----")
                    print("Paths:", '\n' + surface_save_path + '\n' + part_pc_save_path + '\n' + occupancy_save_path)
                    print("Size of GT files for last mesh:", size / 10 ** 6, "M")
                    print("Average size:", np.mean(np.array(sizes)), "M")
                    print("Max size:", np.max(np.array(sizes)), "M")
                    print("Total size:", np.sum(np.array(sizes)), "M")


if __name__ == '__main__':
    t0 = time.time()

    # Parser
    parser = argparse.ArgumentParser(description='Script to generate occupancy and surface coverage gain ground truth '
                                                 'data from ShapeNetCore.v1 meshes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "generate_shapenet_data_config.json".')

    args = parser.parse_args()

    if args.config:
        json_name = args.config
    else:
        json_name = 'generate_shapenet_data_config.json'

    # Load config file and generate data
    print("Using config file:")
    print(json_name)

    json_path = os.path.join(configs_dir, json_name)
    params = load_params(json_path)
    main(params)
    print("Generation finished after", time.time() - t0, "seconds.")
