import imageio as iio
import cv2
import numpy as np
import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Create videos with background removed using median subtraction and optional binarization"
    )
    parser.add_argument("--vid", required=False, type=str, default=None)
    parser.add_argument("--ims", required=False, type=str, default=None)
    parser.add_argument("--fps", required=False, type=int, default=None)
    parser.add_argument("--save_to",
                        required=False,
                        type=str,
                        default="output")
    parser.add_argument("--med_n", required=False, type=int, default=3)
    parser.add_argument("--bin",
                        required=False,
                        action="store",
                        type=int,
                        default=None,
                        metavar='')
    parser.add_argument("--bin_otsu", required=False, action="store_true")
    parser.add_argument("--bin_trng", required=False, action="store_true")
    parser.add_argument("--save_in", required=False, action="store_true")
    parser.add_argument("--out_v", required=False, action="store_true")
    parser.add_argument("--out_f", required=False, action="store_true")
    return parser.parse_args()


def estimate_background(frames):
    return np.median(frames, axis=0).astype(dtype=np.uint8)


def checkpath(path):
    if os.path.exists(path):
        pass
    else:
        print("Error - file not found or invalid path, quitting")
        quit()


def from_video(path):
    filename = path.split("/")[-1]
    print(f"Source video file: {filename}")
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames, fps, filename.split(".")[0]


def from_frames(path):
    contents = os.listdir(path)
    dirname = path.split("/")[-1]
    print(f"Source frame directory: {dirname}")
    contents.sort(key=lambda x: int(x.split(".")[0]))
    frames = []
    for file in contents:
        fpath = os.path.join(path, file)
        try:
            frames.append(cv2.imread(fpath))
        except:
            print(f"Error when reading file {fpath} as image")
    return frames, dirname


def save_video(frames, fps, is_color, fname, path):
    fpath = os.path.join(path, f"{fname}.avi")
    if os.path.exists(fpath):
        n = 1
        while os.path.exists(fpath):
            fpath = os.path.join(path, f"{fname}{n}.avi")
            n += 1

    size = frames[0].shape[1], frames[0].shape[0]
    output = cv2.VideoWriter(fpath, cv2.VideoWriter_fourcc(*"MJPG"), fps, size,
                             is_color)
    print("Saving...")
    for frame in frames:
        output.write(frame)
    output.release()
    print(f"Successfully saved video file {fpath}")


def save_frames(frames, fname, path):
    dir_path = os.path.join(path, f"{fname}")
    if os.path.exists(dir_path):
        n = 1
        while os.path.exists(dir_path):
            dir_path = os.path.join(path, f"{fname}{n}")
            n += 1
    os.makedirs(dir_path)
    n = 0
    print("Saving...")
    for frame in frames:
        frame_ind = str(n).zfill(5)
        fpath = os.path.join(dir_path, f"{frame_ind}.png")
        iio.imwrite(fpath, frame, "PNG")
        sys.stdout.write(f"\rSaved frame {n+1} of {len(frames)}")
        sys.stdout.flush()
        n += 1
    print(f"Successfully saved frames to: {dir_path}")


def main():
    is_color = True
    args = parse_args()

    if args.vid is None and args.ims is None:
        print("Error - no inputs specified, quitting")
        quit()

    if args.out_v is False and args.out_f is False:
        print("Error - no outputs specified, quitting")
        quit()

    if not os.path.exists(args.save_to):
        os.makedirs(args.save_to)

    if args.vid is not None:
        checkpath(args.vid)
        src_frames, fps, instance_name = from_video(args.vid)

    elif args.ims is not None:
        checkpath(args.ims)
        src_frames, instance_name = from_frames(args.ims)
    
        if args.fps is not None and args.out_v is True:
            print(f"Output video will be saved with specified fps: {args.fps}")
            fps = args.fps
            if args.save_in:
                print("Saving original video from frames")
                save_video(src_frames, fps, True, "input_from_frames",
                            args.save_to)
        else:
            print("\nForgot to specify FPS")
            args.out_v, args.out_f = False, True
            
    out_name = f"{instance_name}_bg_remove"
    
    if args.out_f:
        print("Output will be saved as individual frames")

    # estimate initial background

    images = []
    for i in range(args.med_n):
        images.append(src_frames[i])
    bg = estimate_background(images)
    out_frames = []
    n = 0

    # calculate images with background removed

    for frame in src_frames:
        if n >= args.med_n:
            prev_frames = np.array(
                list(src_frames[y] for y in range(n, n - args.med_n, -1)))
            bg = estimate_background(prev_frames)
        out_frame = cv2.absdiff(frame, bg)
        out_frames.append(out_frame)
        sys.stdout.write(f"\rProcessed frame {n+1} of {len(src_frames)}")
        sys.stdout.flush()
        n += 1

    # check whether to binarize

    if args.bin is not None or args.bin_otsu or args.bin_trng:
        is_color = False
        if args.bin_otsu:
            out_frames = [
                cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_OTSU)[1] for frame in out_frames
            ]

        elif args.bin_otsu:
            out_frames = [
                cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 255,
                              cv2.THRESH_TRIANGLE)[1] for frame in out_frames
            ]

        elif args.bin is not None:
            if args.bin < 0 or args.bin > 255:
                print(
                    "\n Error - specified threshold not within an 8-bit pixel intesity value, saving in color RGB"
                )
                is_color = True
            else:
                out_frames = [
                    cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                  args.bin, 255, cv2.THRESH_BINARY)[1]
                    for frame in out_frames
                ]

    if not is_color:
        print("\nOutput successfully binarized")

    print(f"\nSaving to: {args.save_to}")
    if args.out_f:
        save_frames(out_frames, out_name, args.save_to)
    elif args.out_v:
        save_video(out_frames, fps, is_color, out_name, args.save_to)


if __name__ == '__main__':
    main()
