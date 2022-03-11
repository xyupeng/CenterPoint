from ..registry import DETECTORS
from .single_stage import SingleStageDetector
from copy import deepcopy 

@DETECTORS.register_module
class PointPillars(SingleStageDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(PointPillars, self).__init__(
            reader, backbone, neck, bbox_head, train_cfg, test_cfg, pretrained
        )

    def extract_feat(self, data):
        input_features = self.reader(data["features"], data["num_voxels"], data["coors"])
        # input_features: shape=(batch_tot_voxels, 64)

        x = self.backbone(  # scatter as a sparse pseudo image (BEV)
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )  # shape=(bsz, 64, 468, 468)
        if self.with_neck:
            x = self.neck(x)  # shape=(bsz, 384, 468, 468)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]  # tensor(shape=(batch_tot_voxels, 20, 5))
        coordinates = example["coordinates"]  # tensor(shape=(batch_tot_voxels, 4))
        num_points_in_voxel = example["num_points"]  # tensor(shape=(batch_tot_voxels,))
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],  # ndarray([468, 468, 1])
        )

        x = self.extract_feat(data)  # shape=(bsz, 384, 468, 468)
        preds, _ = self.bbox_head(x)
        # preds: [{'reg', 'height', 'dim', 'rot', 'hm'}]
        # _: tensor(shape=(3, 64, 468, 468))

        if return_loss:
            return self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def forward_two_stage(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )

        x = self.extract_feat(data)
        bev_feature = x 
        preds, _ = self.bbox_head(x)

        # manual deepcopy ...
        new_preds = []
        for pred in preds:
            new_pred = {} 
            for k, v in pred.items():
                new_pred[k] = v.detach()

            new_preds.append(new_pred)

        boxes = self.bbox_head.predict(example, new_preds, self.test_cfg)

        if return_loss:
            return boxes, bev_feature, self.bbox_head.loss(example, preds, self.test_cfg)
        else:
            return boxes, bev_feature, None 
