import argparse
import shutil
import os

def clean():
    path = './data'
    shutil.rmtree(path)
    os.mkdir(path)

if __name__ == '__main__':
    clean()