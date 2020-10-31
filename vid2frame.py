import os
import sys
import h5py
import numpy as np
import argparse
import subprocess
import json
from PIL import Image

from subprocess import call

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="the input file path")
parser.add_argument("output_path", type=str, help="the output file path")
# resize options
parser.add_argument("-a", "--asis", action="store_true", help="do not resize frames")
parser.add_argument("-s", "--short", type=int, default=0, help="keep the aspect ration and scale the shorter side to s")
parser.add_argument("-H", "--height", type=int, default=0, help="the resized height")
parser.add_argument("-W", "--width", type=int, default=0, help="the resized width")
# frame sampling options
parser.add_argument("-k", "--skip", type=int, default=1, help="only store frames with (ID-1) mod skip==0, frame ID starts from 1")
parser.add_argument("-n", "--num_frame", type=int, default=-1, help="uniformly sample n frames, this will override --skip")
parser.add_argument("-r", "--interval", type=int, default=0, help="extract one frame every r seconds")

args = parser.parse_args()

def get_frame_rate(vid):
    call = ["ffprobe","-v", "quiet", "-show_entries", "stream=r_frame_rate", "-print_format", "json", vid]
    output = subprocess.check_output(call)
    output = json.loads(output)
    r_frame_rate = 0
    if len(output.keys()) == 0:
        return r_frame_rate
    elif output['streams'] == []:
        return r_frame_rate

    for line in output['streams']:
        nums = line['r_frame_rate'].split('/')
        if float(nums[1]) == 0:
            continue
        frame_rate = 1.0*float(nums[0]) / float(nums[1])
        if frame_rate != 0:
            r_frame_rate = frame_rate

    return r_frame_rate

def load_image(filename) :
    img = Image.open(filename, 'r')
    img.load()
    data = np.asarray( img, dtype="uint8" )
    return data

# sanity check of the options
if args.asis:
    assert args.short == 0 and args.height == 0 and args.width == 0
if args.short > 0:
    assert (not args.asis) and args.height == 0 and args.width == 0
if args.height > 0 or args.width > 0:
    assert (not args.asis) and args.height > 0 and args.width > 0 and args.short == 0

tmp_dir = '/tmp'
vid = args.input_path
vvid, _ = os.path.splitext(vid) # discard extension
_, vvid = os.path.split(vvid)   # get filename without path
v_dir = os.path.join(tmp_dir, vvid)
call(["rm", "-rf", v_dir])
os.mkdir(v_dir)    # caching directory to store ffmpeg extracted frames

if args.asis:
    vf_scale = []
elif args.short > 0:
    vf_scale = ["-vf",
                "scale='iw*1.0/min(iw,ih)*%d':'ih*1.0/min(iw,ih)*%d'" \
                        % (args.short, args.short)]
elif args.height > 0 and args.width > 0:
    vf_scale = ["-vf", "scale=%d:%d" % (args.width, args.height)]
else:
    raise Exception('Unspecified frame scale option')

if args.interval > 0:
    r_frame_rate = get_frame_rate(vid)
    if r_frame_rate == 0:
        print("frame rate is 0, skip: %s"%vid)
    vf_sample = ["-vsync","vfr",
                  "-vf","select=not(mod(n\,%d))" % (int(round(args.interval*r_frame_rate)))]
    assert args.num_frame <= 0 and args.skip == 1, \
            "No other sampling options are allowed when --interval is set"
else:
    vf_sample = []

call(["ffmpeg",
        "-loglevel", "panic",
        "-i", vid,
        ]
        + vf_scale
        + vf_sample
        +
        [
        "-qscale:v", "2",
        v_dir+"/%8d.jpg"])

sample = (args.num_frame > 0)
if sample:
    ids = [int(f.split('.')[0]) for f in os.listdir(v_dir)]
    sample_ids = set(list(np.linspace(min(ids), max(ids),
                            args.num_frame, endpoint=True, dtype=np.int32)))

files = []
for f_name in os.listdir(v_dir):
    fid = int(f_name.split('.')[0])

    if sample:
        if fid not in sample_ids:
            continue
    elif args.skip > 1:
        if (fid-1) % args.skip != 0:
            continue
    files.append((fid, f_name))

call(["rm", "-rf", args.output_path])

with h5py.File(args.output_path, 'w') as output_file: # write mode
    for index, (fid, f_name) in enumerate(files):
        group_name = "%08d" % (fid)   # by padding zeros, frames in db are stored in order
        group = output_file.create_group(group_name)
        image = load_image(os.path.join(v_dir, f_name))
        group.create_dataset('data', image.shape, dtype='uint8', data=image)

call(["rm", "-rf", v_dir])
