from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.colors import Normalize
from PIL import Image
import cv2
from typing import Union, List
import os
from tqdm import tqdm

class CameraData:
    """Class for storing and visualizing RGB and depth image dataset"""
    
    def __init__(self,
                 timestamps: Union[List, np.ndarray] = None, 
                 color_images: Union[List, np.ndarray] = None,
                 depth_images: Union[List, np.ndarray] = None,
                 load_data_from_files: bool = True):
        
        # Load input times
        if timestamps is not None:
            self.timestamps = np.array(timestamps)
        
        # Load RGB images
        self.color = self.ImageInfo()
        if color_images is not None:
            if isinstance(color_images[0], str):  # dir or files
                self.color.data = self._check_image_files(color_images)

                if load_data_from_files:
                    self.color.data = self._load_image_files(self.color.data)
            else:  # numeric
                self.color.data = np.array(
                    [np.array(img) for img in color_images]
                )
        
        # Load depth images
        self.depth = self.ImageInfo()
        if depth_images is not None:
            if isinstance(depth_images[0], str):  # dir or files
                self.depth.data = self._check_image_files(depth_images)

                if load_data_from_files:
                    self.depth.data = self._load_image_files(self.depth.data)
            else:  # numeric
                self.depth.data = np.array(
                    [np.array(img) for img in depth_images]
                )

        # Set extra attribute to hold special (normalized) depth
        self.ndepth = self.ImageInfo()
        if self.depth.data is not None:
            self.ndepth.data = self.normalize_depth(method='global')
                
        # Validate data consistency
        self._validate_data()

        # Initialize MultiPlotter
        self.multiplotter = None
    
    @property
    def elapsed_times(self):
        return self.timestamps - self.timestamps[0]

    @dataclass
    class ImageInfo:
        data: Union[List, np.ndarray] = None
        plotter = None
        
    def _check_image_files(self, paths):
        """Verify image files exist or directory contains images"""

        # Enforce list
        if isinstance(paths, str):
            paths = [paths]

        # Load image contents inside directory if provided directory
        if len(paths) == 1 and os.path.isdir(paths[0]):
            contents = os.listdir(paths[0])
            exts     = [os.path.splitext(file)[1] for file in contents]
            is_image = [True if ext in Image.registered_extensions() else False
                        for ext in exts]
            paths    = [os.path.join(paths[0], file) 
                        for (file, v) in zip(contents, is_image) if v]

        # Check files
        for path in paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f'Image file not found: {path}')

        return paths
    
    def _load_image_files(self, contents) -> np.ndarray:
        """Load image data from file"""
        image_data = []

        for path in contents:
            with Image.open(path) as img:
                image_data.append(np.array(img))

        return np.array(image_data)  # (N, H, W) or (N, H, W, C)
    
    def normalize_depth(self, method = 'global', data = None, dmin = None,
                        dmax = None) -> np.ndarray:
        """
        Returns image data normalized globally over set or by image.

        Args:
            method: Normalize globally ('global') or by image ('image')
            data: Image data (defaults to self.depth.data)
            dmin: Override minimum value for normalization
            dmax: Override maximum value for normalization
        """

        # Get image data
        if data is None:
            data = self.depth.data
        
        if isinstance(data[0], str):  # must load from file
            files = self._check_image_files(data)
            data  = self._load_image_files(files)  # (N, H, W)

        # Get min and max for each image
        match method.lower():
            case 'global':
                images_min = np.nanmin(data) * np.ones(data.__len__())
                images_max = np.nanmax(data) * np.ones(data.__len__())
            case 'image':
                images_min = np.nanmin(data, axis=(1,2))
                images_max = np.nanmax(data, axis=(1,2))
            case _:
                raise ValueError(f"Unknown input for 'method': {method}")
            
        # Apply threshold overrides
        if dmin:
            images_min[images_min < dmin] = dmin
        if dmax:
            images_max[images_max > dmax] = dmax

        # Perform normalization (norm 'uint16' or 'uint8' -> 'float32')
        new_data = np.zeros_like(data, dtype='float32')
        for (ii, image) in enumerate(data):
            norm = Normalize(vmin=images_min[ii], vmax=images_max[ii])
            new_data[ii] = norm(image)

        return new_data
    
    def _validate_data(self):
        """Validate consistent lengths"""
        lengths = []
        checked = []
        attributes = ['timestamps', 'color', 'depth', 'ndepth']

        for attr, value in self.__dict__.items():
            if attr not in attributes:
                continue
            if value is not None:
                lengths.append(len(value.data))
                checked.append(attr)

        if set(lengths).__len__() != 1:
            print('Mismatch in data length:')
            for u, v in zip(checked, lengths):
                print(f'\t{u} = {v}')
            raise ValueError
    
    def plot_color(self, colormap = None):
        """Return ImagePlotter object for RGB images"""
        if self.color is None:
            raise ValueError("No RGB images available")
        
        self.color.plotter = ImagePlotter(self.color.data, self.timestamps)

        if colormap is not None:
            self.color.plotter.colormap = colormap

        return self.color.plotter
    
    def plot_depth(self, colormap = None, **kwargs):
        """Return ImagePlotter object for 'depth'"""
        if not self.depth:
            raise ValueError("No depth images available")
            
        if kwargs:
            self.depth.data = self.normalize_depth(**kwargs)

        self.depth.plotter = ImagePlotter(self.depth.data, self.timestamps)

        if colormap is not None:
            self.depth.plotter.colormap = colormap

        return self.depth.plotter
    
    def plot_ndepth(self, colormap = None, **kwargs):
        """Return ImagePlotter object for 'ndepth'"""
        if not self.depth:
            raise ValueError("No depth images available")
        
        if kwargs:
            self.ndepth.data = self.normalize_depth(**kwargs)
        else:
            self.ndepth.data = self.normalize_depth()

        self.ndepth.plotter = ImagePlotter(self.depth.data, self.timestamps)

        if colormap is not None:
            self.ndepth.plotter.colormap = colormap

        return self.ndepth.plotter
    
    def create_video(self, plotter, savefile: str, method = 'anim', 
                     colormap = None, **kwargs):
        """
        Create and save a video from image sequences.
        
        Methods:
            'anim': (default) matplotlib.animation.FuncAnimation
            'cv2':  cv2.VideoWriter
        """
        plt.ioff()  # turn off interactive mode

        # Check input type
        if not isinstance(plotter, ImagePlotter):
            raise TypeError(f'Bad input type for video: {type(plotter)}')
        
        # Set colormap
        if colormap is not None:
            plotter.colormap = colormap
        
        # Generate video according to method
        match method.lower():
            case 'anim':
                self.create_video_anim(plotter, savefile, **kwargs)
            case 'cv2':
                self.create_video_cv2(plotter, savefile, **kwargs)
            case _:
                raise ValueError(f"Unknown input for 'method': {method}")

        plt.ion()  # turn interactive mode back on
    
    def create_video_cv2(self, plotter, savefile: str, fps: float = 20,
                         realtime: bool = False):

        # Do real-time by adjusting number of times to write each frame
        if realtime:
            dt       = np.diff(plotter.elapsed_times)
            avg_dt   = np.mean(dt)
            avg_fps  = 1/avg_dt
            duration = np.hstack([dt, avg_dt])  # last frame uses avg_dt

            if (avg_fps < fps):
                num_write = np.maximum(1, fps*duration)
            else:  # avg_fps >= fps
                print(f'Average FPS ({avg_fps:.1f}) >= {fps:.1f} FPS.'
                      f' Resetting FPS to {avg_fps:.1f}.')
            
            fps = avg_fps
            num_write = np.ones(np.size(duration), dtype=int)
        else:  # not realtime
            num_write = np.ones(plotter.__len__(), dtype=int)

        # Calculate video dimensions
        figsize = plotter.fig.get_size_inches()
        dpi     = plotter.fig.get_dpi()
        fig_width_px  = int(figsize[0] * dpi)
        fig_height_px = int(figsize[1] * dpi)
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(savefile, 
                              fourcc, 
                              fps, 
                              (fig_width_px, fig_height_px), 
                              isColor=True
                             )
        
        # Generate video
        num_frames = plotter.__len__()
        try:
            for ii in tqdm(range(num_frames), desc='Writing video frames'):

                # Update plot
                plotter.index = ii

                # Get frame data from buffer
                frame = plotter.get_frame()
                
                # Write frame to video
                for _ in range(num_write[ii]):
                    out.write(frame)
                
        finally:
            out.release()
            print(f"Video saved to: {savefile}")

    def create_video_anim(self, plotter, savefile: str, fps: float = 20,
                          realtime: bool = False):
        """Create and save video using Matplotlib animation (faster)"""

        # Do real-time by using average time step, assumed near constant
        if realtime:
            dt     = np.diff(plotter.elapsed_times)
            avg_dt = np.mean(dt)
            std_dt = np.std(dt)
            coef   = std_dt / avg_dt

            if 0.2 >= coef > 0.1:
                print('Warning: std(dt)/mean(dt) = {coef:.2f} exceeds 10%')
            elif coef > 0.2:
                raise ValueError(
                    'Failed assumption of near-constant time interval:'
                    f' std(dt)/mean(dt) = {coef:.2f}'
                )

            avg_fps = 1/avg_dt
            fps = np.uint8(np.round(avg_fps))
            print(f'Using real-time FPS: {fps}')

        # Set up progress bar
        progress_bar = tqdm(total=plotter.__len__(), desc='Rendering video')
                            
        # Define helper animate function
        plotter.show()  # need to assure ax.images[0] is available
        plotter.autoupdate = False  # draw with __animate() instead
        def _animate(idx):
            plotter.index = idx

            plotter.ax.images[0].set_array(plotter.data[idx])  # image
            
            if plotter.title is None:
                frame_num = plotter.index + 1
                num_frame = plotter.__len__()
                text = f'Frame {frame_num}/{num_frame}'

                t_avail = plotter.timestamps
                if t_avail is not None and (len(t_avail) != 0):
                    dt = plotter.elapsed_times[plotter.index]
                    text += f' | Elapsed: {dt:.2f} s'

                plotter.ax.set_title(text, animated=True)
            
            progress_bar.update()
        
            return [plotter.ax.images[0], plotter.ax.title]
        
        # Ensure animation is updated with current colormap
        if plotter.colormap is not None:
            plotter.ax.images[0].set_cmap(plotter.colormap)
        
        # Create animation with blitting for performance
        anim = FuncAnimation(
            plotter.fig,
            _animate,
            frames=plotter.__len__(),
            interval=1000/fps,  # milliseconds btwn frames
            blit=True,
            repeat=False
        )
        
        # Generate video
        try:
            # Set up writer
            writer = FFMpegWriter(fps=fps)

            # Save animation
            try:
                anim.save(savefile, writer=writer)
            except FileNotFoundError as err:
                print(err)
                raise FileNotFoundError(
                    "Need to install 'ffmpeg'."
                    " Execute: `sudo apt install ffmpeg`."
                )
                
        finally:
            progress_bar.close()
            print(f'Video saved to: {savefile}')

    def __len__(self):
        return len(self.timestamps)
    
class ImagePlotter:
    def __init__(self, data, timestamps = None, title = None, colormap = None):
        self.data = data
        self.timestamps = timestamps
        self.fig, self.ax = plt.subplots()
        self.ax.axis('off')
        self.title = title
        self.colormap = colormap
        self.autoupdate = True
        self._index = 0

    @property
    def elapsed_times(self):
        return self.timestamps - self.timestamps[0]
        
    @property
    def index(self):
        if self._index < 0:
            return len(self.data) + self._index
        else:
            return self._index
    
    @index.setter
    def index(self, idx):
        if -len(self.data) <= idx < len(self.data):
            self._index = idx
        else:
            raise IndexError(f'index out of range {len(self.images)} images')
        if self.autoupdate:
            self.update_plot()
        
    def show(self):
        """Shows figure"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            self.ax.axis('off')
        
        self.fig.show()
        self.update_plot()
        return self
    
    def update_plot(self):
        if self.colormap:
            self.ax.imshow(self.data[self._index], cmap=self.colormap)
        else:
            self.ax.imshow(self.data[self._index])

        if self.title is None:
            frame_num = self.index + 1
            num_frame = self.__len__()
            text = f'Frame {frame_num}/{num_frame}'

            if self.timestamps is not None and (len(self.timestamps) != 0):
                dt = self.elapsed_times[self._index]
                text += f' | Elapsed: {dt:.2f} s'

            self.ax.set_title(text)
        else:
            self.ax.set_title(self.title)

        self.fig.canvas.draw()

    def get_frame_old(self):
        if self.fig is None:
            raise ValueError('No figure from which to fetch frame data')
                
        # Get RGBA buffer
        rgba_buffer = np.frombuffer(
            self.fig.canvas.tostring_rgb(),
            dtype=np.uint8  # 8-bit rgb
        )

        # Reshape one-dimensional buffer into image format (H, W, C)
        image_shape = self.fig.canvas.get_width_height()[::-1] + (3,)
        rgba_buffer = rgba_buffer.reshape(image_shape)
                
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(rgba_buffer, cv2.COLOR_RGB2BGR)

        return frame
    
    def get_frame(self):
        """
        Gets image buffer using updated matplotlib.

        Looks like version 3.10 (or 3.7?) did away with .tostring_rgb().
        From documentation and GitHub source code, FigureCanvasAgg has
        .buffer_rgba() and .print_to_buffer().  The latter also returns
        the renderer's dimensions:

            print_to_buffer() -> (bytes(renderer.buffer_rgba()),
                                  (int(renderer.width),
                                   int(renderer.height
                                  )
                                 )
        """

        if self.fig is None:
            raise ValueError('No figure from which to fetch frame data')
                
        # Get buffer and dimensions
        (buffer, image_shape) = self.fig.canvas.print_to_buffer()
        buffer = np.frombuffer(  # from byte-like object to array
            buffer,
            dtype=np.uint8  # 8-bit rgb
        )
        image_shape = image_shape[::-1] + (3,)  # (w, h) -> (h, w, C)

        # Reform original image effect in format (H, W, C)
        r = buffer[0::4]  # red
        g = buffer[1::4]  # green
        b = buffer[2::4]  # blue
        rgb_buffer = np.stack([r, g, b], axis=-1).reshape(image_shape)
                
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(rgb_buffer, cv2.COLOR_RGB2BGR)

        return frame

    def __len__(self):
        return len(self.data)
    
if __name__ == '__main__':
    # --- CONFIG ---
    LOAD_CSV = True
    RUN_DIR  = os.path.join('/mnt/kwu/datasets/wayfaster/',
                            'realsense/data_valid/',
                            'ts_merged_2021_11_09_16h15m31s/')
    SAVE_PIC = False
    SAVE_VID = False

    # --- PRE-PROCESSING ---
    if LOAD_CSV:
        from read_csv_with_vectors import read_csv_with_vectors

    images_csv = os.path.join(RUN_DIR, 'images.csv')

    df = read_csv_with_vectors(images_csv)
    timestamps   = df.timestamp.to_numpy()
    color_images = df.image.to_list()
    depth_images = df.depth.to_list()
    print(f'Loaded {len(timestamps)} data points'
          f' from {os.path.basename(images_csv)}')
    
    color_images = [os.path.join(RUN_DIR, img) for img in color_images]
    depth_images = [os.path.join(RUN_DIR, img) for img in depth_images]
    
    # --- MAIN ---
    camera = CameraData(timestamps=timestamps,
                        color_images=color_images,
                        depth_images=depth_images)
    
    if SAVE_PIC:
        color = camera.plot_color()
        color.show()
        print('Initialized figure for RGB image')
        color.index = -1
        color.fig.savefig(SAVE_PIC)
        print(f'Saved RGB image to {SAVE_PIC}')
    
    if SAVE_VID:
        color = camera.plot_color()
        camera.create_video(color, 'sample_1.mp4')

        depth = camera.plot_depth()
        depth.colormap = 'gray'
        camera.create_video(depth, 'sample_2.mp4')