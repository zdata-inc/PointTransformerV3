from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import open3d as o3d
import torch.nn.functional as F

from PointTransformer.datasets.transform import *
from PointTransformer.datasets.utils import collate_fn
from PointTransformer.models.builder import build_model
from PointTransformer.utils.config import Config

ROOT_DIR = Path('/home/asakhare/github/external/PointTransformerV3/Pointcept')
EXP_DIR = Path(ROOT_DIR, 'exp/s3dis/semseg-pt-v3m1-1-ppt-extreme')
CFG_FILEPATH = r'/home/asakhare/github/external/PointTransformerV3/PointTransformer/configs/semseg-pt-v3m1-1-ppt-extreme.py'
WEIGHTS_FILEPATH = Path(EXP_DIR, 'model/model_best.pth')
PCD_FILEPATH = Path('/home/asakhare/data/mw/store_downsampled_.05.ply')
SAVE_FILEPATH = Path(EXP_DIR, f'result/{PCD_FILEPATH.stem}_pred.npy')

def run_inference(model, data_dict, cfg):
    fragment_list = data_dict.pop("fragment_list")
    segment = data_dict.pop("segment")

    pred = torch.zeros((segment.size, cfg.data.num_classes)).cuda()

    for i in range(len(fragment_list)):
        fragment_batch_size = 1
        s_i, e_i = i * fragment_batch_size, min((i + 1) * fragment_batch_size, len(fragment_list))
        input_dict = collate_fn(fragment_list[s_i:e_i])

        for key in input_dict.keys():
            if isinstance(input_dict[key], torch.Tensor):
                input_dict[key] = input_dict[key].cuda(non_blocking=True)

        idx_part = input_dict["index"]
        with torch.no_grad():
            pred_part = model(input_dict)["seg_logits"]  # (n, k)
            pred_part = F.softmax(pred_part, -1)

            bs = 0
            for be in input_dict["offset"]:
                pred[idx_part[bs:be], :] += pred_part[bs:be]
                bs = be

    pred = pred.max(1)[1].data.cpu().numpy()

    return pred

def get_data(pcd_filepath, cfg):
    cfg = cfg.data
    pcd = o3d.io.read_point_cloud(pcd_filepath)
    points = np.asarray(pcd.points).astype(np.float32)
    colors = (np.asarray(pcd.colors) * 255).astype(np.float32)
    segment = np.ones(points.shape[0]) * -1

    transform = Compose(cfg.test.transform)
    test_voxelize = (TRANSFORMS.build(cfg.test.test_cfg.voxelize))
    post_transform = Compose(cfg.test.test_cfg.post_transform)
    aug_transform = [Compose(aug) for aug in cfg.test.test_cfg.aug_transform]

    data_dict = {'coord': points, 'color': colors}
    data_dict = transform(data_dict)

    data_dict_list = []
    for aug in aug_transform:
        data_dict_list.append(aug(deepcopy(data_dict)))

    fragment_list = []
    for data in data_dict_list:
        data_part_list = test_voxelize(data)

        for data_part in data_part_list:
            data_part = [data_part]
            fragment_list += data_part

    for i in range(len(fragment_list)):
        fragment_list[i] = post_transform(fragment_list[i])

    data_dict = {'fragment_list': fragment_list, 'segment': segment}

    return data_dict


def load_weights(model, weights_filepath):
    checkpoint = torch.load(weights_filepath)
    weight = OrderedDict()
    for key, value in checkpoint["state_dict"].items():
        if key.startswith("module."):
            key = key[7:]  # module.xxx.xxx -> xxx.xxx
        weight[key] = value

    model.load_state_dict(weight, strict=True)

    return model

cfg = Config.fromfile(CFG_FILEPATH)

model = build_model(cfg.model)
model = load_weights(model, WEIGHTS_FILEPATH)
model.to(device='cuda:0')

data_dict = get_data(PCD_FILEPATH.as_posix(), cfg)

pred = run_inference(model, data_dict, cfg)

print(np.unique(pred))


np.save(SAVE_FILEPATH, pred)