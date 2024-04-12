import glob
from pathlib import Path

import numpy as np
import open3d as o3d

PRED_DIR = r'C:\Users\AshwinSakhare\MyDrive\GitHub\External\PointTransformerV3\Pointcept\exp\s3dis\semseg-pt-v3m1-1-ppt-extreme\result'
PCD_DIR = r'C:\Users\AshwinSakhare\MyDrive\GitHub\zData\koda_wayfinding\data'

LABEL_COLORS = {0: [1, 0, 0],  # Red, ceiling
                1: [0, 1, 0],  # Green, floor
                2: [0, 0, 1],  # Blue, wall
                3: [1, 1, 0],  # Yellow, beam
                4: [1, 0, 1],  # Magenta, column
                5: [0, 1, 1],  # Cyan, window
                6: [0.5, 0.5, 0],  # Olive, door
                7: [0.5, 0, 0.5],  # Purple, table
                8: [0, 0.5, 0.5],  # Teal, chair
                9: [0.5, 0.5, 0.5],  # Gray, sofa
                10: [0.75, 0.25, 0],  # Orange, bookcase
                11: [0.75, 0, 0.25],  # Dark Red, board
                12: [0, 0, 0]  # Black, clutter
                }

LABEL_COLORS = {0: [1, 0, 0],  # Red, ceiling
                1: [0, 1, 0],  # Green, floor
                2: [0, 0, 1],  # Blue, wall
                12: [0, 0, 0]  # Dark Blue, clutter
                }

pcd_filepaths = glob.glob(PCD_DIR + "/" + 'store_downsampled_.05.ply')

# Iterate over each file in the directory
for filename in pcd_filepaths:
    name = Path(filename).stem
    suffix = Path(filename).suffix
    pcd_filepath = Path(PCD_DIR, filename).as_posix()
    pred_filepath = Path(PRED_DIR, name + '_pred.npy')

    # Read the point cloud
    pcd = o3d.io.read_point_cloud(pcd_filepath)

    # Load predictions
    preds = np.load(pred_filepath)

    colors = [LABEL_COLORS[pred] if pred in LABEL_COLORS else LABEL_COLORS[12] for pred in preds]

    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors))

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])
