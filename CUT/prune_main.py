import argparse
import os
import sys
import torch
import torch.nn as nn
from mmcv import Config
from mmcv.runner import load_checkpoint

sys.path.insert(0, './experiments/CUT')


def parse_args():
    parser = argparse.ArgumentParser(description='Prune ViT-MoE model')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--save-weights', required=True, help='output weight path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--resume-from', help='resume from checkpoint')
    args = parser.parse_args()
    return args


def prune_mlp_hidden_dims(state_dict, mlp_hidden_dims, original_mlp_ratio=4):
    embed_dim = 768
    new_state_dict = {}

    for key, value in state_dict.items():
        if 'blocks' in key and 'mlp' in key:
            parts = key.split('.')
            blocks_idx = parts.index('blocks')
            block_idx = int(parts[blocks_idx + 1])

            if 'gate' in key:
                new_state_dict[key] = value
            elif 'experts' in key:
                target_dim = mlp_hidden_dims[block_idx]
                if 'weight' in key:
                    new_state_dict[key] = value[:, :target_dim].clone()
                elif 'bias' in key:
                    new_state_dict[key] = value
                else:
                    new_state_dict[key] = value
            elif 'fc1.weight' in key:
                original_dim = embed_dim * original_mlp_ratio
                target_dim = mlp_hidden_dims[block_idx]
                new_state_dict[key] = value[:target_dim, :].clone()
            elif 'fc1.bias' in key:
                target_dim = mlp_hidden_dims[block_idx]
                new_state_dict[key] = value[:target_dim].clone()
            elif 'fc2.weight' in key:
                original_dim = embed_dim * original_mlp_ratio
                target_dim = mlp_hidden_dims[block_idx]
                new_state_dict[key] = value[:, :target_dim].clone()
            elif 'fc2.bias' in key:
                new_state_dict[key] = value.clone()
            else:
                new_state_dict[key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def extract_backbone_state_dict(full_state_dict):
    backbone_state_dict = {}
    for key, value in full_state_dict.items():
        if key.startswith('backbone.'):
            new_key = key[len('backbone.'):]
            backbone_state_dict[new_key] = value
    return backbone_state_dict


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    print(f"Loading base model from: {cfg.model.pretrained}")
    checkpoint = torch.load(cfg.model.pretrained, map_location='cpu')

    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    new_mlp_hidden_dims = cfg.get('mlp_hidden_dims', None)
    if new_mlp_hidden_dims is None:
        print("No mlp_hidden_dims specified in config, skipping prune.")
        backbone_dict = extract_backbone_state_dict(state_dict)
        torch.save(backbone_dict, args.save_weights)
        return

    print(f"Pruning MLP hidden dims from [3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072, 3072] to {new_mlp_hidden_dims}")

    pruned_state_dict = prune_mlp_hidden_dims(state_dict, new_mlp_hidden_dims)
    
    backbone_state_dict = extract_backbone_state_dict(pruned_state_dict)

    os.makedirs(os.path.dirname(args.save_weights), exist_ok=True)
    torch.save(backbone_state_dict, args.save_weights)
    print(f"Pruned backbone weights saved to: {args.save_weights}")


if __name__ == '__main__':
    main()