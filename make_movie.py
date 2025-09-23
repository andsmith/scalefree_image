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
"""
import time
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
COLORS = {'text': (51, 4, 0),
          'bkg': (227, 238, 246)}  # BGR

test_data_linear = {'frame_rate': 30,
                    'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
                    'title_pause_sec': 5.0,
                    'episode_initial_pause_sec': 1.0,
                    'episode_final_pause_sec': 3.0,
                    'txt_color': COLORS['text'],
                    'bkg_color': COLORS['bkg'],
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


test_data_circular = {'frame_rate': 30,
                      'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
                      'title_pause_sec': 5.0,
                      'episode_initial_pause_sec': 1.0,
                      'episode_final_pause_sec': 3.0,
                      'txt_color': COLORS['text'],
                      'bkg_color': COLORS['bkg'],
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


test_data_barn = {'frame_rate': 20,
                  'caption_height_px': 50,  # shrink/grow if if you need more/fewer lines of text
                  'title_pause_sec': 10.0,
                  'episode_initial_pause_sec': .5,
                  'episode_final_pause_sec': 5.0,
                  'txt_color': COLORS['text'],
                  'bkg_color': COLORS['bkg'],
                  'title_pad_px': 30,
                  'caption_pad_xy': (10, 5),
                  'title': {'main': ('Optimal image approximation using:',
                                     '  experiment 1:  50 lines  ',
                                     '  experiment 2:  50 circles',
                                     '  experiment 3:  25 of each.'),
                            'sub1': ('1 frame = 50 epochs',),
                            'sub2': ('by:  Andrew T. Smith, 2025',
                                     'github: andsmith/scalefree_image',),
                            'spacing_frac': 0.2,
                            'max_font_scales': (None, 1.0, None)
                            },
                  'train_img': {'file': 'barn_train_25c-25l_20h_downscale=6.0.png',
                                'caption': ['training image (99 x 54)'],
                                'duration_sec': 5.0,
                                'inter-episode_pause_sec': 3.0},
                  'episodes': [
                      {'input_pattern': r'movie\\barn_linear_50d_20h_cycle-????????.png',
                       'caption': ['50 line units, 20 color units']},
                      {'input_pattern': r'movie\\barn_circular_50d_20h_cycle-????????.png',
                       'caption': ['50 circle units, 20 color units']},
                      {'input_pattern': r'movie_mix\\barn_output_25c-25l_20h_cycle-????????.png',
                       'caption': ['25 line units + 25 circle units, 20 color units',
                                        'epochs@LR: 250@1.0, 500@0.1, 500@0.01']},
                  ]}

test_data_barn_meta = {'frame_rate': 10,
                       'caption_height_px': 80,  # shrink/grow if if you need more/fewer lines of text
                       'title_pause_sec': 5.0,
                       'episode_initial_pause_sec': 2.0,
                       'episode_final_pause_sec': 4.0,
                       'txt_color': COLORS['text'],
                       'bkg_color': COLORS['bkg'],
                       'title_pad_px': 30,
                       'caption_pad_xy': (10, 5),
                       'title': {'main': ('Optimal image approximation using',
                                          ' 25 circle units + 25 line units,',
                                          ' 20 color units.'),
                                 'sub1': tuple(),
                                 'sub2': ('By:  Andrew T. Smith, 2025',
                                          'github: andsmith/scalefree_image',),
                                 'spacing_frac': 0.2,
                                 'max_font_scales': (None, 1.2, None)
                                 },
                       'train_img': {'file': 'barn_train_20c-20l_20h_downscale=3.0.png',
                                     'caption': ['training image (198 x 109)'],
                                     'duration_sec': 5.0},
                       'episodes': [
                           {'json_meta': r'barn_metadata_25c-25l_20h.json',
                            'caption': [{'txt': 'cycle: %d', 'meta_keys': ['cycle']},
                                        {'txt': 'learning rate: %.5f', 'meta_keys': ['learning_rate']},
                                        {'txt': 'loss: %.7f', 'meta_keys': ['loss']}]}
                       ]
                       }
still_life = {'frame_rate': 10,
              'caption_height_px': 80,  # shrink/grow if if you need more/fewer lines of text
              'title_pause_sec': 5.0,
              'episode_initial_pause_sec': 2.0,
              'episode_final_pause_sec': 4.0,
              'txt_color': COLORS['text'],
              'bkg_color': COLORS['bkg'],
              'title_pad_px': 30,
              'caption_pad_xy': (10, 5),
              'max_frame_cap_font_scale': 1.2,
              'title': {'main': ('Optimal image approximation using',
                                 ' 50 circle units + 100 line units,',
                                 ' 50 color units.'),
                        'sub1': ('Image by: Clara Peeters, ca. 1610,',),
                        'sub2': ('Code by:  Andrew T. Smith, 2025',
                                 'github: andsmith/scalefree_image',),
                        'spacing_frac': 0.2,
                        'max_font_scales': (None, 1.2, None)
                        },
              'train_img': {'file': 'still_life_train_50c-100l_32h_downscale=4.0.png',
                            'caption': ['input data: %d x %d'],
                            'duration_sec': 5.0},
              'episodes': [
                  {'json_meta': r'still_life_metadata_50c-100l_32h.json',
                   'caption': [{'txt': 'cycle (50 epochs): %d, learning rate: %.5f', 'meta_keys': ['cycle', 'learning_rate']},
                               {'txt': 'output %i x %i, loss: %.7f', 'meta_keys': ['train_width', 'train_height', 'loss']}]}
              ]
              }


class MovieMaker(object):
    def __init__(self, movie_data, output_file, preview=False):
        self.movie_data = movie_data
        self.output_file = output_file
        self.preview = preview
        if 'train_img' in self.movie_data:
            self.train_img = cv2.imread(self.movie_data['train_img']['file'])[:, :, ::-1]
            logging.info("Loaded training image of shape:  %s" % (self.train_img.shape,))
        else:
            self.train_img = None
        self.frame_rate = self.movie_data['frame_rate']
        self.caption_height_px = self.movie_data['caption_height_px']
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
        self._max_frame_cap_font_scale = self.movie_data.get('max_frame_cap_font_scale', 1.0)
        # self.episodes = self._load()

    def make_train_frame(self, size_wh):
        if self.train_img is None:
            raise ValueError("No training image provided in movie data.")
        train_img = self.train_img[:, :, ::-1]
        if '%' in self.movie_data['train_img']['caption'][0]:  # TODO better way to detect this
            caption_top = self.movie_data['train_img']['caption'][0] % (train_img.shape[1], train_img.shape[0])
        else:
            caption_top = self.movie_data['train_img']['caption'][0]
        if (train_img.shape[1], train_img.shape[0]) != size_wh:
            logging.warning(f"Resizing training image from ({train_img.shape[1]}, {train_img.shape[0]}) to {size_wh}")
            train_img = cv2.resize(train_img, size_wh, interpolation=cv2.INTER_AREA)
        captions = [caption_top] + self.movie_data['train_img']['caption'][1:]
        frame = captioned_frame(train_img, captions, self.caption_height_px, self.caption_pad_xy, max_font_scale=self._max_frame_cap_font_scale,
                                txt_color=self.txt_color, bkg_color=self.bkg_color, font_face=cv2.FONT_HERSHEY_COMPLEX, line_spacing=1.5)
        return frame

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
        add_text(frame, self.title_txt['main'], main_bbox, font_face=cv2.FONT_HERSHEY_DUPLEX,
                 justify='left', color=self.txt_color, line_spacing=1.5, **kwargs)

        if len(self.title_txt['sub1']) > 0:
            kwargs = {} if max_font_scales[1] is None else {'max_font_scale': max_font_scales[1]}
            add_text(frame, self.title_txt['sub1'], sub1_bbox, line_spacing=1.5,
                     font_face=cv2.FONT_HERSHEY_COMPLEX, justify='left', color=self.txt_color, **kwargs)
        if len(self.title_txt['sub2']) > 0:
            kwargs = {} if max_font_scales[2] is None else {'max_font_scale': max_font_scales[2]}
            add_text(frame, self.title_txt['sub2'], sub2_bbox, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                     justify='left', line_spacing=1.5, color=self.txt_color, **kwargs)
        return frame

    def run(self):
        frames = self._make_frame_sequence()
        if self.preview:
            while not self._preview(frames):
                logging.info("Restarting preview, hit 'q' or ESC in the preview window to exit.")
        if self.output_file is not None:
            # Check all frames are the same size
            frame_size = frames[0].shape[1], frames[0].shape[0]
            for f_i, f in enumerate(frames):
                print(f.shape)
                if (f.shape[1], f.shape[0]) != frame_size:
                    logging.warning("Frame size mismatch:  %s should be %s, resizing..." %
                                    ((f.shape[1], f.shape[0]), frame_size))
                    frames[f_i] = cv2.resize(f, frame_size, interpolation=cv2.INTER_AREA)
            self._write_movie(frames)

    def _preview(self, frames):
        t0 = time.perf_counter()
        last_time = t0 - 10.0
        n_frames = 0
        max_delay = 1.0 / self.frame_rate
        sleep_times = []
        user_quit = False
        for i, frame in enumerate(frames):
            now = time.perf_counter()
            delay = now - last_time
            sleep_time = max_delay - delay

            sleep_times.append(sleep_time)

            if sleep_time > 0:
                time.sleep(sleep_time)
            cv2.imshow('preview', frame)
            n_frames += 1
            last_time = time.perf_counter()

            if n_frames % 30 == 0:
                elapsed = now - t0
                actual_fps = n_frames / elapsed if elapsed > 0 else 0
                logging.info(f"Preview frame {i}, actual fps: {actual_fps:.2f}")
                t0 = now
                n_frames = 0

            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                user_quit = True
                break
        if user_quit:
            logging.info("User requested exit from preview.")
            cv2.destroyAllWindows()
            return True
        return False

    def _load_episode_frames(self, episode):
        frames = []
        if 'input_pattern' in episode:
            files = glob.glob(episode['input_pattern'])
            if not files:
                logging.error(f"No files found for episode with pattern:  {episode['input_pattern']}")
                return []
            n_digits = episode['input_pattern'].count('?')
            files = self._sort_frame_files(files, n_digits)
            meta = None
            logging.info("Loaded %d filenames for episode with pattern:  %s" % (len(files), episode['input_pattern']))
        elif 'json_meta' in episode:
            with open(episode['json_meta'], 'r') as f:
                meta = json.load(f)
            meta = self._sort_metadata(meta)
            files = [m['filename'] for m in meta if 'filename' in m]
            logging.info("Metadata file:  %s, loaded %d frame entries." % (episode['json_meta'], len(meta)))
            if not files:
                logging.error(f"No filenames found in metadata file:  {episode['json_meta']}")
                return []
        else:
            raise ValueError("Episode must have either 'input_pattern' or 'json_meta' key.")

        for f_i, f in enumerate(files):
            img = cv2.imread(f)
            if img is None:
                raise ValueError(f"Failed to read image file for frame {f_i}:  {f}")
            frames.append(img)

        if meta is not None:
            # Some keys need to be extracted from the frames, get those now
            for m, f in zip(meta, frames):
                m['train_width'] = f.shape[1]
                m['train_height'] = f.shape[0]

        if not frames:
            raise ValueError(f"No valid image files found for episode with pattern:  {episode['input_pattern']}")
        logging.info(f"Loaded {len(frames)} frames for episode.")
        return frames, meta

    def _make_frame_caption(self, meta, base_caption):
        """
        :param meta: metadata dictionary to take values for the caption from.
        :param base_captions: list of dictionaries, one per line of caption, each with keys:
               'txt': a format string with %d, %f, etc. for values from 'meta'
               'meta_keys': list of keys to get values from the 'meta' dict.
        """
        lines = []
        for cap in base_caption:
            fmt = cap['txt']
            keys = cap['meta_keys']
            values = [meta[k] if k in meta else 'N/A' for k in keys]
            try:
                line = fmt % tuple(values)
            except (TypeError, KeyError) as e:
                logging.error(f"Error formatting caption line: {e}")
                line = fmt % ('N/A',) * len(keys)
            lines.append(line)
        return lines

    def _make_episode_seq(self, episode):
        frames, meta = self._load_episode_frames(episode)
        if meta is None:
            frame_captions = [episode['caption']] * len(frames)
        else:
            # Need to make custom captions from the metadata for each frame
            frame_captions = [self._make_frame_caption(m, episode['caption']) for m in meta]

        frames = [captioned_frame(f, c, self.caption_height_px, self.caption_pad_xy, justify='left', max_font_scale=self._max_frame_cap_font_scale,
                                  txt_color=self.txt_color, bkg_color=self.bkg_color, line_spacing=1.5) for f, c in zip(frames, frame_captions)]

        intro_frames = self._mk_seq(frames[0], self.initial_pause_sec)
        outro_frames = self._mk_seq(frames[-1], self.final_pause_sec)
        frames = intro_frames + frames + outro_frames

        return frames

    def _make_frame_sequence(self):
        episode_sequences = [self._make_episode_seq(ep) for ep in self.episode_data]
        frame_size_wh = episode_sequences[0][0].shape[:2][::-1]
        img_size = (frame_size_wh[0], frame_size_wh[1]-self.caption_height_px)
        logging.info("Got image/frame size from first episode's first frame:  %s / %s" % (img_size, frame_size_wh))
        title_frame = self.make_title_frame(frame_size_wh,  # include caption area
                                            spacing_frac=self._title_spacing_frac,
                                            max_font_scales=self._max_title_font_scales)

        frames = self._mk_seq(title_frame, self.title_dur_sec,)
        short_train_seq = []
        if self.train_img is not None:
            train_frame = self.make_train_frame(img_size)
            train_seq = self._mk_seq(train_frame, self.movie_data['train_img']['duration_sec'])
            short_train_seq = self._mk_seq(
                train_frame,  self.movie_data['train_img'].get('inter-episode_pause_sec', 2.0))

            frames += train_seq
        for seq_no, seq in enumerate(episode_sequences):
            frames += seq + (short_train_seq if (seq_no < len(episode_sequences)-1) else [])
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

    def _sort_frame_files(self, files, n_digits):
        if not files:
            return []
        # Extract the number from the filename, it will be where n_digits '?' are in the pattern
        regex_pattern = r'(\d{' + str(n_digits) + r'})'
        regex = re.compile(regex_pattern)

        def extract_number(f):
            match = regex.search(f)
            if match:
                return int(match.group(1))
            else:
                return float('inf')  # If no match, put it at the end

        files.sort(key=extract_number)
        return files

    def _sort_metadata(self, meta):
        """
        assume filename is formatted ...000324.png,
        i.e. ending with the frame number with leading zeros, then an extension.
        Sort the metadata list by this number.
        """

        n_digits = len(re.search(r'([0-9]+)\.[a-zA-Z0-9]', meta[0]['filename']).group(1))
        regex_pattern = r'(\d{' + str(n_digits) + r'})'
        regex = re.compile(regex_pattern)

        def extract_number(m):
            match = regex.search(m['filename'])
            if match:
                return int(match.group(1))
            else:
                return float('inf')  # If no match, put it at the end
        meta.sort(key=extract_number)
        return meta


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

        movie_data = deepcopy(still_life)
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
        train_frame = movie_maker.make_train_frame((800, 600))
        cv2.imwrite('example_title.png', title_frame[:, :, ::-1])
        logging.info("No output mp4 file provided, wrote title frame to: example_title.png")
        cv2.imwrite('example_train.png', train_frame[:, :, ::-1])
        logging.info("Wrote training image frame to: example_train.png")
        sys.exit(0)

    if args.play:
        logging.info("Playing movie preview...")
        movie_maker = MovieMaker(movie_data, None, preview=True)
    elif args.output_file is not None:
        logging.info(f"Creating movie file:  {args.output_file}")
        movie_maker = MovieMaker(movie_data, args.output_file, preview=False)
    movie_maker.run()
