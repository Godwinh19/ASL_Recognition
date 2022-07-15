import torch
import torch.nn as nn

from torchvision import transforms
import videotransforms

import numpy as np

from pytorch_i3d import InceptionI3d
from datasets.nslt_dataset import load_rgb_frames_from_video, video_to_tensor
from decoder import get_gloss

NUM_CLASSES = 2000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = './checkpoints/nslt_2000_065846_0.447803.pt'

VIDEO_ROOT = "../../../asl_video"


def pad(imgs, total_frames=64):
    if imgs.shape[0] < total_frames:
        num_padding = total_frames - imgs.shape[0]

        if num_padding:
            prob = np.random.random_sample()
            if prob > 0.5:
                pad_img = imgs[0]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
            else:
                pad_img = imgs[-1]
                pad = np.tile(np.expand_dims(pad_img, axis=0), (num_padding, 1, 1, 1))
                padded_imgs = np.concatenate([imgs, pad], axis=0)
    else:
        padded_imgs = imgs

    return padded_imgs


# ----------------* VIDEO PROCESSING *-------------------
imgs = load_rgb_frames_from_video(vid_root=VIDEO_ROOT, vid="above", start=0, num=64)

test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
imgs = pad(imgs)

# Run through the data augmentation
# 64 x 224 x 224 x 3
imgs = test_transforms(imgs)
ret_img = video_to_tensor(imgs)
inputs = ret_img[np.newaxis, ...]
print(inputs.shape)

# ----------------* END VIDEO PROCESSING *-----------------


i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load('weights/rgb_imagenet.pt'))

i3d.replace_logits(NUM_CLASSES)
i3d.load_state_dict(torch.load(
    weights,
    map_location=device))  # nslt_2000_000700.pt nslt_1000_010800 nslt_300_005100.pt(best_results)  nslt_300_005500.pt(results_reported) nslt_2000_011400
i3d = nn.DataParallel(i3d)
i3d.eval()

per_frame_logits = i3d(inputs)
## 1 x num_classes
predictions = torch.max(per_frame_logits, dim=2)[0]
# predictions[0] --> num_classes tensor
# lowest as the first element - highest as the last element
out_labels = np.argsort(predictions.cpu().detach().numpy()[0])
out_probs = np.sort(predictions.cpu().detach().numpy()[0])

gloss = get_gloss(out_labels[-1])
print("gloss predicted: ", gloss)
