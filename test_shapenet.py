import argparse
from macarons.testers.shapenet import *

dir_path = os.path.abspath(os.path.dirname(__file__))
test_configs_dir = os.path.join(dir_path, "./configs/test/")

if __name__ == '__main__':
    # Parser
    parser = argparse.ArgumentParser(description='Script to test occupancy probability and surface coverage gain '
                                                 'modules on ShapeNetCore.v1 meshes.')
    parser.add_argument('-c', '--config', type=str, help='name of the config file. '
                                                         'Default is "test_on_shapenet_seen_categories_config.json".')

    args = parser.parse_args()

    if args.config:
        params_name = args.config
    else:
        params_name = "test_on_shapenet_seen_categories_config.json"

    params_name = os.path.join(test_configs_dir, params_name)
    test_params = load_params(params_name)

    with torch.no_grad():
        run_test(test_params)
