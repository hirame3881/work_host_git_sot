print("hello")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-testint', type=int)
args=parser.parse_args()
for i in range(args.testint):
    print("hi")