"""
Assemble generated frames into a movie using ffmpeg.
Read a json describing the movie parameters.

File structure:

   {'frame_rate': fps int,
    'title_duration_sec': float,  # seconds to hold the title frame
    'title': { 'main': (line1, line2, ...),  # title text lines
               'sub1': (line1, line2, ...),  # subtitle text lines
               'sub2': (line1, line2, ...)}, # subtitle text lines
    'caption_height_px': int,  # black, below the image
    'episodes': [ ...]
    'initial_pause_sec': float, seconds to hold the first frame
    "final_duration_sec": float, seconds to hold the final frame
   }

Each episode is a dictionary with keys:
    "input_pattern": A glob pattern for the input frames, e.g. "frames/frame_?????.png".
                    (NOTE, must be the correct number of ?'s for the digits.)
                    The ? will be replaced with digits starting from 0, frames sorted by this number.
    "caption": a text caption to display below the image
    "resolution": (width, height)  # optional, if present and not (0, 0), resize frames to this resolution
         otherwise use the frame's native resolution (and issue warnings if they differ)
"""
import os
import subprocess
import logging
import sys
from copy import deepcopy
import argparse
import json
from tempfile import tempdir
from util import add_text, draw_bbox, captioned_frame
import cv2
import glob
import re
import numpy as np

test_data = {'frame_rate': 15,
             'resolution': (0, 0), # use first image size
             'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
             'title_pause_sec': 2.0,
             'episode_initial_pause_sec': 1.5,
             'episode_final_pause_sec': 3.0,
             'txt_color': (255, 255, 255),
             'bkg_color': (0, 0, 0),
             'title_pad_px': 30,
             'caption_pad_xy': (10, 5),
             'title': {'main': ('Cut the plane with N lines',
                                'Color each region to',
                                'approximate an image.'),
                       'sub1': ('N:  8, 32, 128, 512          ',),
                       'sub2': ('By: Andrew T. Smith, 2025      ',
                                'github:andsmith/scalefree_image')
                       },
             'episodes': [
                 {'input_pattern': r'movies\\washington_linear_8d_10h_cycle-????????.png',
                  'caption': ['8 linear dividers', '10 hidden units']},
                 {'input_pattern': r'movies\\washington_linear_32d_10h_cycle-????????.png',
                  'caption': ['32 linear dividers', '10 hidden units']},
                 {'input_pattern': r'movies\\washington_linear_128d_32h_cycle-????????.png',
                  'caption': ['128 linear dividers', '32 hidden units']},
                 {'input_pattern': r'movies\\washington_linear_512d_64h_cycle-????????.png',
                  'caption': ['512 linear dividers', '64 hidden units']}
             ]
             }


class MovieMaker(object):
    def __init__(self, movie_json, output_file, preview=False):
        self.movie_json = movie_json
        self.output_file = output_file
        self.preview = preview

        with open(movie_json, 'r') as f:
            self.movie_data = json.load(f)

        self.frame_rate = self.movie_data['frame_rate']
        self.caption_height_px = self.movie_data['caption_height_px']
        self.resolution = tuple(self.movie_data['resolution'])
        self.episode_data = self.movie_data['episodes']
        self.title_txt = self.movie_data['title']
        self.title_pad_px = self.movie_data['title_pad_px']
        self.caption_pad_xy = self.movie_data['caption_pad_xy']
        self.initial_pause_sec = self.movie_data['episode_initial_pause_sec']
        self.final_pause_sec = self.movie_data['episode_final_pause_sec']
        self.title_dur_sec = self.movie_data['title_pause_sec']
        self.txt_color = tuple(self.movie_data['txt_color'])
        self.bkg_color = tuple(self.movie_data['bkg_color'])
        #self.episodes = self._load()

    def _make_title_frame(self, size_wh, spacing_frac=0.2):
        """
        +-------------------+
        |   title line 1    |
        |   title line 2    |
        |   title line 3    |
        |                   |
        |    sub1 line 1    |
        |    sub1 line 2    |
        |                   |
        |    sub2 line 1    |
        |    sub2 line 2    |
        +-------------------+
        :param size_wh: (width, height) of the image area
        :param pad_px: padding in pixels
        :param spacing_frac: fraction of the image height to use for spacing (between text line groups)
        """
        pad_px = self.title_pad_px
        text_height = size_wh[1]-2*pad_px
        pad_height = int(spacing_frac*text_height) // 2
        text_width = size_wh[0]-2*pad_px

        # Determine relative heights of each text box
        n_lines = [len(self.title_txt['main']) + 1, len(self.title_txt['sub1']), len(self.title_txt['sub2'])] # +1 for main title sizing
        total_lines = sum(n_lines)
        if total_lines == 0:
            total_lines = 1
        # Compute relative heights
        rel_heights = [nh / total_lines for nh in n_lines]
        rel_heights = [rh * (text_height - pad_height * (len(n_lines) - 1)) for rh in rel_heights]
        box_heights = [int(rh) for rh in rel_heights]

        main_bbox = {'x':(pad_px, pad_px+text_width),
                     'y':(pad_px, pad_px + box_heights[0])}
        y_top = pad_px + box_heights[0] + pad_height
        sub1_bbox = {'x':(pad_px, pad_px+text_width),
                     'y':(y_top, y_top + box_heights[1])}
        y_top = y_top + box_heights[1] + pad_height
        sub2_bbox = {'x':(pad_px, pad_px+text_width),
                     'y':(y_top, y_top + box_heights[2])}
        
        frame = np.zeros((size_wh[1]+self.caption_height_px, size_wh[0], 3), dtype=np.uint8)
        frame[:, :] = self.bkg_color
        
        add_text(frame, self.title_txt['main'], main_bbox,font_face = cv2.FONT_HERSHEY_COMPLEX, font_thickness=2, justify='center',color=self.txt_color)
        add_text(frame, self.title_txt['sub1'], sub1_bbox, font_face = cv2.FONT_HERSHEY_SIMPLEX, justify='center', color=self.txt_color)
        add_text(frame, self.title_txt['sub2'], sub2_bbox, font_face = cv2.FONT_HERSHEY_SIMPLEX, justify='left', line_spacing=2.5, color=self.txt_color)
        return frame
    
    def run(self):
        frames = self._make_frame_sequence()
        if self.preview:
            self._preview(frames)
        if self.output_file is not None:
            self._write_movie(frames)
    def _preview(self, frames):
        for i, frame in enumerate(frames):
            cv2.imshow('preview', frame)
            key = cv2.waitKey(int(1000/self.frame_rate))
            if key == 27:  # ESC
                break
        cv2.destroyAllWindows()
    def _load_episode_frames(self, episode):
        files = self._get_files(episode['input_pattern'])
        if not files:
            logging.error(f"No files found for episode with pattern:  {episode['input_pattern']}")
            return []
        frames = []
        for f in files:
            img = cv2.imread(f)
            if img is None:
                logging.warning(f"Could not read image file:  {f}")
                continue
            if self.resolution != (0, 0):
                if img.shape[1] != self.resolution[0] or img.shape[0] != self.resolution[1]:
                    img = cv2.resize(img, self.resolution, interpolation=cv2.INTER_AREA)
            else:
                if frames and (img.shape[1] != frames[0].shape[1] or img.shape[0] != frames[0].shape[0]):
                    logging.warning(f"Image {f} has different size ({img.shape[1]}, {img.shape[0]}) than previous images ({frames[0].shape[1]}, {frames[0].shape[0]}).")
            frames.append(img)
        if not frames:
            raise ValueError(f"No valid image files found for episode with pattern:  {episode['input_pattern']}")
        logging.info("Loaded %d frames for episode with pattern:  %s" % (len(frames), episode['input_pattern']))
        return frames

    def _make_episode_seq(self, episode):
        frames = self._load_episode_frames(episode)
        frame_captions = episode['caption']
        frames = [captioned_frame(f, frame_captions, self.caption_height_px, self.caption_pad_xy, txt_color=self.txt_color, bkg_color=self.bkg_color) for f in frames]
        intro_frame = frames[0]
        outro_frame = frames[-1]
        frames = self._mk_seq(intro_frame, self.initial_pause_sec) + frames + self._mk_seq(outro_frame, self.final_pause_sec)
        
        return frames

    def _make_frame_sequence(self):
        episode_sequences = [self._make_episode_seq(ep) for ep in self.episode_data]
        frame_size_wh = episode_sequences[0][0].shape[:2][::-1]
        title_frame = self._make_title_frame(frame_size_wh)
        frames = self._mk_seq(title_frame, self.title_dur_sec)
        for seq in episode_sequences:
            frames += seq
        return frames
    
    def _mk_seq(self, frame, dur_sec):
        n_frames = int(dur_sec * self.frame_rate)
        return [frame] * n_frames
    
    def _write_movie(self, frames):
        cmd = ['ffmpeg',
               '-y',  # overwrite output file if it exists
               '-framerate', str(self.frame_rate),
               '-i', 'pipe:',  # input from stdin
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               self.output_file]
        logging.info("Running ffmpeg command:  %s" % (' '.join(cmd),))
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i, frame in enumerate(frames):
            logging.debug(f"Writing frame {i}")
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait()
        if proc.returncode != 0:
            logging.error(f"ffmpeg exited with code {proc.returncode}")
        else:
            logging.info(f"Movie written to {self.output_file}")

    def _get_files(self, glob_pattern):
        """
        Get a sorted list of files matching the glob pattern.
        The pattern should include '?' characters for the digits, e.g. 'frame_?????.png'.
        """
        files = glob.glob(glob_pattern)
        if not files:
            return []
        # Extract the number from the filename using regex
        digit_count = glob_pattern.count('?')
        regex_pattern = re.escape(glob_pattern).replace(r'\?' * digit_count, r'(\d{' + str(digit_count) + r'})')
        regex = re.compile(regex_pattern)
        
        def extract_number(f):
            match = regex.search(f)
            if match:
                return int(match.group(1))
            else:
                return float('inf')  # If no match, put it at the end
        
        files.sort(key=extract_number)
        return files    
    
def get_args():
    parser = argparse.ArgumentParser(description="Make a movie from frames")
    parser.add_argument('movie_json', type=str, help="Json file describing the movie")
    parser.add_argument('output_file', type=str, help="Output movie file, e.g. movie.mp4")
    parser.add_argument('-p', '--play', action='store_true', help="Play the movie, don't create it.", default=False)
    parser.add_argument('-e', '--example', action='store_true', help="Write example JSON movie file.", default=False)

    return parser.parse_args()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    if args.example:
        example = deepcopy(test_data)
        
        with open('example_movie.json', 'w') as f:
            json.dump(example, f, indent=4)
        logging.info("Wrote example_movie.json")
        sys.exit(0)
    output_file = args.output_file if args.play is False else None
    movie_maker = MovieMaker(args.movie_json, output_file, preview=args.play)
    frame = movie_maker._make_title_frame((800, 600))

    print(frame.shape)
    cv2.imwrite('title_frame.png', frame)

    movie_maker.run()
    logging.info(f"Movie saved to:  {args.output_file}")