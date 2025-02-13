import torch


def create_rollout_anchors_mask(boxes, scores, images_input_size, backbone_input_size, score_thr, device):
    anchors_mask = torch.zeros(backbone_input_size[0], backbone_input_size[1], dtype=torch.bool, device=device)
    valid_boxes = torch.greater(scores, score_thr)
    if torch.any(valid_boxes):
        boxes = boxes[valid_boxes, :]
        ds_factor = backbone_input_size[0] / images_input_size[0]
        ds_boxes = boxes * ds_factor
        ds_boxes[:, :2] = torch.floor(ds_boxes[:, :2])
        ds_boxes[:, 2:] = torch.ceil(ds_boxes[:, 2:])
        ds_boxes = torch.clip(ds_boxes, 0.0, backbone_input_size[0] - 1).int()
        for box in ds_boxes:
            anchors_mask[box[1]:box[3], box[0]:box[2]] = True
    else:
        anchors_mask[:] = True
    anchors_mask = torch.flatten(anchors_mask)
    return anchors_mask
