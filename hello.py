#import tensorflow
import os
print(os.path.basename(__file__)+"a")

import argparse    # 1. argparseをインポート
parser = argparse.ArgumentParser(description='このプログラムの説明（なくてもよい）')   
parser.add_argument('--sort_flag', action='store_true')
parser.add_argument('--mI_supple_type', type=int,default=0)
args=parser.parse_args()

print(args.sort_flag)
print(args.mI_supple_type)
if args.mI_supple_type:
    print("true")
print(int(0.999))