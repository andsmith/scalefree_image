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
from util import add_text, captioned_frame
import cv2
import glob
import re
import numpy as np

test_data_linear = {'frame_rate': 20,
                    'resolution': (0, 0),  # use first image size
                    'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
                    'title_pause_sec': 5.0,
                    'episode_initial_pause_sec': 1.0,
                    'episode_final_pause_sec': 3.0,
                    'txt_color': (255, 255, 255),
                    'bkg_color': (0, 0, 0),
                    'title_pad_px': 30,
                    'caption_pad_xy': (10, 5),
                    'title': {'main': ('Cut the plane with N lines,',
                                       'color each region uniformly.',
                                       'Reduce error by adjusting',
                                       'lines to approximate target:',
                                       '   F(x,y) = (R, G, B)'),
                              'sub1': ('N:  8, 32, 128, 512          ',),
                              'sub2': ('By: Andrew T. Smith, 2025      ',
                                       'github:andsmith/scalefree_image'),
                              'spacing_frac': 0.1,

                              },
                    'episodes': [
                        {'input_pattern': r'movies\\washington_linear_8d_10h_cycle-????????.png',
                         'caption': ['8 line units', '10 color units']},
                        {'input_pattern': r'movies\\washington_linear_32d_10h_cycle-????????.png',
                         'caption': ['32 line units', '10 color units']},
                        {'input_pattern': r'movies\\washington_linear_128d_32h_cycle-????????.png',
                         'caption': ['128 line units', '32 color units']},
                        {'input_pattern': r'movies\\washington_linear_512d_64h_cycle-????????.png',
                         'caption': ['512 line units', '64 color units']}
                    ]
                    }


test_data_circular = {'frame_rate': 20,
                      'resolution': (0, 0),  # use first image size
                      'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
                      'title_pause_sec': 5.0,
                      'episode_initial_pause_sec': 1.0,
                      'episode_final_pause_sec': 3.0,
                      'txt_color': (255, 255, 255),
                      'bkg_color': (0, 0, 0),
                      'title_pad_px': 30,
                      'caption_pad_xy': (10, 5),
                      'title': {'main': ('Cut the plane with N circles.',
                                'Color each region uniformly.',
                                         'Pick circle size/position',
                                         'and color intersections to',
                                         'best approximate target.'),
                                'sub1': ('N:  8, 32, 128, 512          ',),
                                'sub2': ('By: Andrew T. Smith, 2025      ',
                                         'github:andsmith/scalefree_image'),
                                'spacing_frac': 0.1,
                                },
                      'episodes': [
                          {'input_pattern': r'movies\\mona_lisa_circular_8d_10h_cycle-????????.png',
                           'caption': ['8 circle units', '10 color units']},
                          {'input_pattern': r'movies\\mona_lisa_circular_32d_16h_cycle-????????.png',
                           'caption': ['32 circle units', '16 color units']},
                          {'input_pattern': r'movies\\mona_lisa_circular_128d_24h_cycle-????????.png',
                           'caption': ['128 circle units', '24 color units']},
                          {'input_pattern': r'movies\\mona_lisa_circular_512d_64h_cycle-????????.png',
                           'caption': ['512 circle units', '64 color units']}
                      ]
                      }
test_data = {'frame_rate': 20,
             'resolution': (0, 0),  # use first image size
             'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
             'title_pause_sec': 2.0,
             'episode_initial_pause_sec': 2.0,
             'episode_final_pause_sec': 4.0,
             'txt_color': (255, 255, 255),
             'bkg_color': (0, 0, 0),
             'title_pad_px': 30,
             'caption_pad_xy': (10, 5),
             'title': {'main': ('Optimal image approximation',
                                'with 50 lines, 50 circles.'),
                       'sub1': ('learn rates: 0.1, 0.01, 0.001,',
                                '            500 epochs each,',
                                '            1 frame = 25 epochs.'),
                       'sub2': ('By:  Andrew T. Smith, 2025',
                                'github: andsmith/scalefree_image',),
                       'spacing_frac': 0.2,
                       'max_font_scales': (None, 1.2, None)
                       },
             'episodes': [
                 {'input_pattern': r'movie\\barn_linear_50d_20h_cycle-????????.png',
                  'caption': ['50 line units', '20 color units']},
                 {'input_pattern': r'movie\\barn_circular_50d_20h_cycle-????????.png',
                     'caption': ['50 circle units', '20 color units']},
                 {'input_pattern': r'movie\\barn_output_15c-15l_20h_cycle-????????.png',
                     'caption': ['15 line units + 15 circle units', '20 color units']}
             ]
             }


class MovieMaker(object):
    def __init__(self, movie_data, output_file, preview=False):
        self.movie_data = movie_data
        self.output_file = output_file
        self.preview = preview

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
        self._title_spacing_frac = self.movie_data['title']['spacing_frac']
        self._max_title_font_scales = self.movie_data['title'].get('max_font_scales', (None, None, None))
        # self.episodes = self._load()

    def make_title_frame(self, size_wh, spacing_frac=0.0, max_font_scales=(None, None, None)):
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
        print("SPACING FRACTION: ", spacing_frac)
        pad_px = self.title_pad_px
        text_height = size_wh[1]-2*pad_px
        pad_height = int(spacing_frac*text_height) // 2
        text_width = size_wh[0]-2*pad_px

        # Determine relative heights of each text box
        n_lines = [len(self.title_txt['main']) + 1, len(self.title_txt['sub1']),
                   len(self.title_txt['sub2'])]  # +1 for main title sizing
        total_lines = sum(n_lines)
        if total_lines == 0:
            total_lines = 1
        # Compute relative heights
        rel_heights = [nh / total_lines for nh in n_lines]
        rel_heights = [rh * (text_height - pad_height * (len(n_lines) - 1)) for rh in rel_heights]
        box_heights = [int(rh) for rh in rel_heights]

        main_bbox = {'x': (pad_px, pad_px+text_width),
                     'y': (pad_px, pad_px + box_heights[0])}
        y_top = pad_px + box_heights[0] + pad_height
        sub1_bbox = {'x': (pad_px, pad_px+text_width),
                     'y': (y_top, y_top + box_heights[1])}
        y_top = y_top + box_heights[1] + pad_height
        sub2_bbox = {'x': (pad_px, pad_px+text_width),
                     'y': (y_top, y_top + box_heights[2])}

        frame = np.zeros((size_wh[1], size_wh[0], 3), dtype=np.uint8)
        frame[:, :] = self.bkg_color
        kwargs = {} if max_font_scales[0] is None else {'max_font_scale': max_font_scales[0]}
        add_text(frame, self.title_txt['main'], main_bbox, font_face=cv2.FONT_HERSHEY_COMPLEX,
                 justify='center', color=self.txt_color, line_spacing=2.5, **kwargs)

        kwargs = {} if max_font_scales[1] is None else {'max_font_scale': max_font_scales[1]}
        add_text(frame, self.title_txt['sub1'], sub1_bbox, line_spacing=2.5,
                 font_face=cv2.FONT_HERSHEY_SIMPLEX, justify='left', color=self.txt_color, **kwargs)
        kwargs = {} if max_font_scales[2] is None else {'max_font_scale': max_font_scales[2]}
        add_text(frame, self.title_txt['sub2'], sub2_bbox, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                 justify='left', line_spacing=2.5, color=self.txt_color, **kwargs)
        return frame

    def run(self):
        frames = self._make_frame_sequence()
        if self.preview:
            self._preview(frames)
        if self.output_file is not None:
            # Check all frames are the same size
            frame_size = frames[0].shape[1], frames[0].shape[0]
            for f in frames:
                if (f.shape[1], f.shape[0]) != frame_size:
                    raise ValueError("All frames must have the same size to write movie, but found differing sizes.")
            self._write_movie(frames)

    def _preview(self, frames):
        for i, frame in enumerate(frames):
            cv2.imshow('preview', frame)
            key = cv2.waitKey(int(1000/self.frame_rate))
            if key == 27 or key == ord('q'):
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
                    logging.warning(
                        f"Image {f} has different size ({img.shape[1]}, {img.shape[0]}) than previous images ({frames[0].shape[1]}, {frames[0].shape[0]}).")
            frames.append(img)
        if not frames:
            raise ValueError(f"No valid image files found for episode with pattern:  {episode['input_pattern']}")
        logging.info("Loaded %d frames for episode with pattern:  %s" % (len(frames), episode['input_pattern']))
        return frames

    def _make_episode_seq(self, episode):
        frames = self._load_episode_frames(episode)
        frame_captions = episode['caption']
        frames = [captioned_frame(f, frame_captions, self.caption_height_px, self.caption_pad_xy,
                                  txt_color=self.txt_color, bkg_color=self.bkg_color) for f in frames]
        intro_frame = frames[0]
        outro_frame = frames[-1]
        frames = self._mk_seq(intro_frame, self.initial_pause_sec) + frames + \
            self._mk_seq(outro_frame, self.final_pause_sec)

        return frames

    def _make_frame_sequence(self):
        episode_sequences = [self._make_episode_seq(ep) for ep in self.episode_data]
        frame_size_wh = episode_sequences[0][0].shape[:2][::-1]
        title_frame = self.make_title_frame(frame_size_wh,
                                            spacing_frac=self._title_spacing_frac,
                                            max_font_scales=self._max_title_font_scales)
        frames = self._mk_seq(title_frame, self.title_dur_sec)
        for seq in episode_sequences:
            frames += seq
        return frames

    def _mk_seq(self, frame, dur_sec):
        n_frames = int(dur_sec * self.frame_rate)
        return [frame] * n_frames

    def _write_movie(self, frames):
        if not frames:
            logging.error("No frames to write")
            return

        # Get frame dimensions from first frame and ensure they're consistent
        first_frame = frames[0]
        height, width, channels = first_frame.shape

        # Ensure all frames have the same dimensions
        for i, frame in enumerate(frames):
            if frame.shape != first_frame.shape:
                logging.warning(f"Frame {i} has different shape {frame.shape}, resizing to {first_frame.shape}")
                frames[i] = cv2.resize(frame, (width, height))

        # Make sure frame dimensions are even (required for yuv420p)
        if width % 2 != 0:
            width -= 1
        if height % 2 != 0:
            height -= 1

        # Resize frames to even dimensions if needed
        if width != first_frame.shape[1] or height != first_frame.shape[0]:
            logging.info(f"Adjusting frame size to {width}x{height} for yuv420p compatibility")
            frames = [cv2.resize(frame, (width, height)) for frame in frames]

        cmd = ['ffmpeg',
               '-y',  # overwrite output file if it exists
               '-f', 'rawvideo',  # specify input format
               '-vcodec', 'rawvideo',
               '-pix_fmt', 'bgr24',  # OpenCV uses BGR format
               '-s', f'{width}x{height}',  # frame size
               '-r', str(self.frame_rate),  # input framerate
               '-i', 'pipe:0',  # input from stdin
               '-c:v', 'libx264',
               '-pix_fmt', 'yuv420p',
               '-r', str(self.frame_rate),  # output framerate
               '-preset', 'medium',  # encoding preset
               self.output_file]

        logging.info("Running ffmpeg command:  %s" % (' '.join(cmd),))
        logging.info(f"Frame dimensions: {width}x{height}, channels: {channels}")

        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, stdout=subprocess.PIPE)

            for i, frame in enumerate(frames):
                if proc.poll() is not None:  # Check if process has terminated
                    logging.error(f"FFmpeg process terminated early at frame {i}")
                    break

                if i % 100 == 0:  # Log progress every 100 frames
                    logging.info(f"Writing frame {i}/{len(frames)}")

                # Ensure frame is contiguous in memory
                if not frame.flags['C_CONTIGUOUS']:
                    frame = np.ascontiguousarray(frame)

                try:
                    frame_bytes = frame.tobytes()
                    expected_size = width * height * channels
                    if len(frame_bytes) != expected_size:
                        logging.error(f"Frame {i} size mismatch: got {len(frame_bytes)}, expected {expected_size}")
                        break
                    proc.stdin.write(frame_bytes)
                except BrokenPipeError:
                    logging.error(f"Broken pipe error at frame {i}")
                    break

            proc.stdin.close()
            stdout, stderr = proc.communicate()

            if proc.returncode != 0:
                logging.error(f"ffmpeg exited with code {proc.returncode}")
                if stderr:
                    logging.error(f"ffmpeg stderr: {stderr.decode()}")
            else:
                logging.info(f"Movie written to {self.output_file}")

        except Exception as e:
            logging.error(f"Error during movie creation: {e}")
            if 'proc' in locals():
                proc.terminate()

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
    parser.add_argument('-m', '--movie_json', type=str,
                        help="Json file describing the movie (uses test_data in make_movie.py if not provided)", default=None)
    parser.add_argument('-o', '--output_file', type=str,
                        help="Output movie file, e.g. movie.mp4, if not provided will just generate a title card.", default=None)
    parser.add_argument('-p', '--play', action='store_true', help="Play the movie, don't create it.", default=False)
    parser.add_argument('-j', '--write_json', type=str,
                        help="Don't make/play a movie, just write the JSON to this file (useful with no -m argument, generate a template.)", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    # load data
    if args.movie_json is None:

        movie_data = deepcopy(test_data)
        logging.info("No movie json file provided, using built-in test data, has %i episodes." %
                     (len(movie_data['episodes']),))
    else:
        with open(args.movie_json, 'r') as f:
            movie_data = json.load(f)
        logging.info(f"Loaded movie json file:  {args.movie_json}, has {len(movie_data['episodes'])} episodes.")

    if args.write_json:
        with open(args.write_json, 'w') as f:
            json.dump(movie_data, f, indent=4)
        logging.info(f"Wrote {args.write_json}")
        sys.exit(0)

    # We need a MovieMaker to generate an mp4, play the movie, and/or make a title card.
    if args.output_file is None and args.play is False:
        movie_maker = MovieMaker(movie_data, None, preview=False)

        # Just make a title card
        # Use 800x600 as a reasonable default size
        title_frame = movie_maker.make_title_frame((800, 600),
                                                   movie_data['title']['spacing_frac'],
                                                   max_font_scales=movie_data['title']['max_font_scales'])
        cv2.imwrite('example_title.png', title_frame)
        logging.info("No output mp4 file provided, wrote title frame to: example_title.png")
        sys.exit(0)

    if args.play:
        logging.info("Playing movie preview...")
        movie_maker = MovieMaker(movie_data, None, preview=True)
    elif args.output_file is not None:
        logging.info(f"Creating movie file:  {args.output_file}")
        movie_maker = MovieMaker(movie_data, args.output_file, preview=False)
    movie_maker.run()
