from PIL import Image
import numpy as np
from typing import Any, Tuple, Dict

from scipy.interpolate import splprep, splev
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit, brentq
from scipy.integrate import quad

from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

import matplotlib.pyplot as plt

import skan
import networkx as nx

__all__ = ['Particle', 'Polydat']

class Particle():
    '''
    Particle object class. Represents a single polymer particle in the full field image.

    Attributes:
        image (np.ndarray):
            The image of the particle.
        bbox (tuple):
            The bounding box of the particle in the full field image.
        skeleton (np.ndarray):
            The skeleton of the particle.
        classification (str):
            The classification of the particle.
    '''
    def __init__(self,
                 image: np.ndarray,
                 resolution: float,
                 bbox: Tuple[int, int, int, int],
                 binary_mask: np.ndarray = None,
                 classification: str = None,
                 particle_id: int = None) -> None:
        '''
        Initialize a Particle object.

        Args:
            image (np.ndarray):
                The image of the particle.
            resolution (float):
                The resolution of the image in nanometers per pixel.
            bbox (tuple):
                The bounding box of the particle in the full field image.
            binary_mask (np.ndarray):
                The binary mask of the particle.
            skeleton (np.ndarray):
                The skeleton of the particle.
            classification (str):
                The classification of the particle.
        '''
        # Set the image attribute.
        self._image = image

        # Set the resolution attribute.
        self._resolution = resolution

        # Set the bounding box attribute.
        self._bbox = bbox

        # Set the binary mask attribute.
        self._binary_mask = binary_mask

        # Set the skeleton attribute.
        self._skeleton = None

        # Set the skeleton summary attribute.
        self._skeleton_summary = None

        # Set the classification attribute.
        self._classification = classification

        # Set the particle id attribute.
        self._particle_id = particle_id

        # Set the contour sampling attribute.
        self._contour_sampling = None

        # Set the contour lengths attribute.
        self._contour_lengths = None

        # Set the interpolated skeleton coordinates attribute.
        self._interp_skeleton_coordinates = None

        # Set the interpolated skeleton derivatives attribute.
        self._interp_skeleton_derivatives = None

        # Set the Tan-Tan correlation attribute.
        self._tantan_correlations = None

    def skeletonize_particle(self,
                             method: str = 'zhang') -> None:
        '''
        Skeletonize the particle's binary mask and set the skeleton attribute.

        Args:
            method (str):
                The method to use for skeletonization. Default is 'lee'. Options are 'lee' and 'zhang'. See the
                documentation for skimage.morphology.skeletonize for more information.
        Returns:
            None
        '''
        skel = skeletonize(self._binary_mask, method = method)
        self._skeleton = skan.Skeleton(skel, source_image = self._image)
        self._skeleton_summary = skan.summarize(self._skeleton, separator = '_')

    def classify(self) -> None:
        '''
        Classify the particle as Linear, Branched, Loop, Branched-Loop, or Unknown based on the skeleton summary.
        
        Args:
            None
        Returns:
            None
        '''
        # Check to see if the skeleton summary is set. If not, raise a ValueError.
        if self._skeleton_summary is None:
            raise ValueError('Skeleton summary attribute is not set. Skeletonize the particle before classifying.')
        
        # Get the unique branch types from the skeleton summary.
        unique_branch_types = self._skeleton_summary['branch_type'].unique()

        # Check if the particle is classified as linear:
        if np.all(unique_branch_types == 0):
            self._classification = 'Linear'

        # Check if the particle is classified as branched-loop:
        elif 1 in unique_branch_types:
            # The particle is some form of branched, now check if it contains any cycles in the graph representation.
            graph = skan.csr.skeleton_to_nx(self._skeleton)
            loops = list(nx.simple_cycles(graph))
            
            # If any loops are found, classify the particle as branched-loop.
            if len(loops) > 0:
                self._classification = 'Branched-Looped'
            else:
                self._classification = 'Branched'

        # Check if the particle is classified as a loop:
        elif np.all(unique_branch_types == 3):
            self._classification = 'Loop'

        # If the particle does not fit any of the above classifications, set the classification to unknown.
        else:
            self._classification = 'Unknown'

    def interpolate_skeleton(self,
                             step_size: float,
                             k: int = 3,
                             s: float = 0.5) -> None:
        '''
        Create an interpolated representation of the skeleton for all paths. This is necessary for calculating the persistence 
        length with subpixel accuracy. Each path is interpolated using a spline of order k with a smoothing factor s.

        Args:
            step_size (float):
                The step size in nanometers along the contour length of the skeleton.
            k (int):
                The degree of the spline. Default is 3.
            s (float):
                The smoothing factor. Default is 0.5.
        Returns:
            None
        '''
        # Check to see if the skeleton is set. If not, raise a ValueError.
        if self._skeleton is None:
            raise ValueError('Skeleton attribute is not set. Skeletonize the particle before interpolating.')

        # Initialize lists to store interpolated values for each path.
        self._contour_samplings = []
        self._contour_lengths = []
        self._interp_skeleton_coordinates = []
        self._interp_skeleton_derivatives = []

        # Loop over all paths in the skeleton.
        num_paths = self._skeleton.n_paths
        for path_idx in range(num_paths):
            # Get the skeleton coordinates for the current path.
            skeleton_coordinates = self._skeleton.path_coordinates(path_idx)

            # Calculate the contour length of the current skeleton path.
            contour_length = self._skeleton.path_lengths()[path_idx]
            
            # This interpolation only works properly for paths with that satisfy the m > k condition in the splprep 
            # function. We attempt to interpolate the path, and if it fails, we skip the path. 
            try:
                # Interpolate the current skeleton path.
                tck, _ = splprep(skeleton_coordinates.T, k=k, s=s)
            except TypeError:
                continue

            # Append the contour length for this path.
            self._contour_lengths.append(contour_length)

            # Set the actual contour coordinates for this path.
            contour_actual = np.arange(0, contour_length, step_size)

            # Set the normalized contour coordinates for this path.
            contour_normalized = contour_actual / contour_length

            # Append the true contour sampling for this path.
            self._contour_samplings.append(contour_actual)

            # Append the interpolated skeleton coordinates for this path.
            self._interp_skeleton_coordinates.append(splev(contour_normalized, tck))

            # Append the interpolated derivative for this path.
            self._interp_skeleton_derivatives.append(splev(contour_normalized, tck, der=1))

    def calc_tantan_correlation(self):
        '''
        Calculate the persistence length of the polymer particle using the Tan-Tan correlation for all skeleton paths. 
        The persistence length is calculated using the formula:
        <cos(theta)> = exp(-L/Lp)
        where L is the contour length of the skeleton, and Lp is the persistence length.

        The persistence length is set to the tantan_pl attribute.

        Args:
            None
        Returns:
            None
        '''
        # Check to see if the interpolated skeleton derivatives are set. If not, raise a ValueError.
        if self._interp_skeleton_derivatives is None:
            raise ValueError('Interpolated skeleton derivatives attribute is not set. Interpolate the skeleton before calculating the persistence length.')

        # Initialize a list to store Tan-Tan correlations for each path.
        self._tantan_correlations = []

        # Loop over each path's interpolated skeleton derivative.
        for  derivative in self._interp_skeleton_derivatives:
            # Calculate the tangent vectors for the current path.
            tangent_vectors = derivative / np.linalg.norm(derivative, axis=0, keepdims=True)

            # Get the number of tangent vectors.
            _, N = tangent_vectors.shape
            # Initialize an array to store the correlation for this path.
            corr = np.zeros(N)

            # lag = 0: Perfect correlation (a vector is always perfectly correlated with itself)
            corr[0] = 1.0
            for k in range(1,N):
                
                # For lag k, consider pairs of vectors separated by k units
                dot_products = np.sum(tangent_vectors[:, :-k] * tangent_vectors[:, k:], axis=0)
                corr[k] = np.mean(dot_products)

            # Append the correlation for this path to the list.
            self._tantan_correlations.append(corr)

    def plot_particle(self, cmap = 'Greys_r'):
        '''
        Plot the particle image.

        Args:
            None
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        # Create the fig and ax.
        fig, ax = plt.subplots(figsize=(4, 4))
        # Set the title
        if self._classification is None:
            ax.set_title(f'Particle {self._particle_id}')
        else:
            ax.set_title(f'Particle {self._particle_id} - {self._classification}')
        # Plot the image.
        ax.imshow(self._image, cmap = cmap)
        # Return the fig and ax.
        return fig, ax
    
    def plot_skeleton(self, cmap = 'Greys_r'):
        '''
        Plot the particle skeleton and interpolated skeleton.
        
        Args:
            None
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        # Create the fig and ax.
        fig, ax = plt.subplots(figsize=(4, 4))
        # Set the title
        if self._classification is None:
            ax.set_title(f'Particle {self._particle_id} Skeleton')
        else:
            ax.set_title(f'Particle {self._particle_id} Skeleton - {self._classification}')

        if self._skeleton is not None:
            # Plot the skeleton
            ax.imshow(self.skeleton.skeleton_image, cmap = cmap, alpha = 0.75)
        else:
            raise ValueError('Skeleton attribute is not set. Skeletonize the particle before plotting.')
        
        # Plot the interpolated skeletons. This only works if the interpolated skeleton coordinates are set.
        if self._interp_skeleton_coordinates is not None:
            for index, (splinex, spliney) in enumerate(self._interp_skeleton_coordinates):
                ax.plot(spliney, splinex, lw=1, label = f'Path {index}')
            ax.legend()
        # if len(self._interp_skeleton_coordinates) > 0:
        #     for index, (splinex, spliney) in enumerate(self._interp_skeleton_coordinates):
        #         ax.plot(spliney, splinex, lw=1, label = f'Path {index}')
        #     ax.legend()
        
        # Return the fig and ax.
        return fig, ax
    
    @property
    def image(self) -> np.ndarray:
        '''
        The image of the particle.
        '''
        return self._image

    @property
    def resolution(self) -> float:
        '''
        The resolution of the image in nanometers per pixel.
        '''
        return self._resolution
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        '''
        The bounding box of the particle in the full field image.
        '''
        return self._bbox
    
    @property
    def binary_mask(self) -> np.ndarray:
        '''
        The binary mask of the particle.
        '''
        return self._binary_mask
    
    @property
    def classification(self) -> str:
        '''
        The classification of the particle.
        '''
        return self._classification
    
    @property
    def particle_id(self) -> int:
        '''
        The unique particle id of the particle.
        '''
        return self._particle_id
    
    @property
    def skeleton(self) -> skan.Skeleton:
        '''
        The skan skeleton of the particle.
        '''
        return self._skeleton
    
    @property
    def skeleton_summary(self) -> 'pandas.DataFrame': # type: ignore
        '''
        The skan skeleton summary of the particle.
        '''
        return self._skeleton_summary
    
    @property
    def contour_samplings(self) -> list[np.ndarray]:
        '''
        The contour sampling of each path in the particle.
        '''
        return self._contour_samplings
    
    @property
    def contour_lengths(self) -> list[float]:
        '''
        The contour lengths of each path in the particle.
        '''
        return self._contour_lengths
    
    @property
    def interp_skeleton_coordinates(self) -> list[np.ndarray]:
        '''
        The interpolated skeleton coordinates of each path in the particle.
        '''
        return self._interp_skeleton_coordinates
    
    @property
    def interp_skeleton_derivatives(self) -> list[np.ndarray]:
        '''
        The interpolated derivative of the skeleton of each path in the particle.
        '''
        return self._interp_skeleton_derivatives
    
    @property
    def tantan_correlations(self) -> list[np.ndarray]:
        '''
        The Tan-Tan correlation of each path in the particle.
        '''
        return self._tantan_correlations
   
class Polydat():
    '''
    Polymer data class. Used for processing a multiple polymer field images. Includes a list of polymer full field images,
    list of polymer particles, and various methods for analysis.

    Attributes:
        images (list):
            List of full field image arrays of the polymer data.
        resolution (float):
            The resolution of the image in nanometers per pixel.
        particles (list):
            List of Particle objects.
        metadata (dict):
            The metadata ssociated with the polymer image. Key-value pairs of metadata.
    '''
    def __init__(self,
                 images: list[np.ndarray] = None,
                 resolution: float = None,
                 **metadata: Any):
        '''
        Initialization method for the Polydat_Multi object.

        Args:
            images list[np.ndarray]:
                The list of full field image arrays of the polymer data.
            resolution (float):
                The resolution of the image in nanometers per pixel.
            metadata (dict):
                The metadata associated with the polymer image. Key-value pairs of metadata.
        Returns:
            None
        '''
        #Set the image attribute.
        self._images = images if images is not None else []

        # Set the resolution attribute.
        self._resolution = resolution

        # Set the metadata attribute.
        self._metadata = metadata

        # Set the particles attribute.
        self._particles = []

        # Set the number of particles attribute.
        self._num_particles = 0

        # Set the contour lengths attribute.
        self._contour_lengths = None

        # Set the contour sampling attribute.
        self._contour_sampling = None

        # Set the mean Tan-Tan correlation attribute.
        self._mean_tantan_correlation = None

        # Set the minimum contour length attribute.
        self._min_contour_length = 0

        # Set the maximum contour length attribute.
        self._max_contour_length = np.inf

        # Set the persistence length attribute.
        self._pl = 0

        # Set the persistence length covariance attribute.
        self._plcov = 0

    @classmethod
    def from_images(cls,
                    filepaths: list[str],
                    resolution: float,
                    **metadata: Any) -> 'Polydat':
        '''
        Create an instance of the Polydat_Multi object from a list of image files.

        Args:
            filepaths (str):
                The list of filepaths to the image files.
            resolution (float):
                The resolution of the image in nanometers per pixel.
            metadata (dict):
                The metadata associated with the polymer image. Key-value pairs of metadata.

        Returns:
            Polydat:
                The Polydat object.
        '''
        images = []
        for filepath in filepaths:
            # Load the image file.
            with Image.open(filepath) as img:
                grayscale = img.convert('L')
                image = np.array(grayscale)/255.0
            images.append(image)
            
        # Create the Polydat object.
        return cls(images = images, resolution = resolution, **metadata)

    def __exp_decay(self, x, Lp):
        '''
        Exponential decay function for curve fitting.

        Args:
            x (float):
                The x value.
            Lp (float):
                The persistence length.
        Returns:
            float:
                The exponential decay value.
        '''
        return np.exp(-x/(2*Lp))
    
    ########################
    ##### Main Methods #####
    ########################

    def add_image(self,
                  filepath: str,
                  resolution: float):
        '''
        Load an image file and add it to the images attribute.

        Args:
            filepath (str):
                The file path to the image file.
            resolution (float):
                The resolution of the image in nanometers per pixel.
        Returns:
            None
        '''
        # Make sure the resolution of the image matches the resolution of the other images.
        if self._resolution is not None and self._resolution != resolution:
            raise ValueError('Resolution mismatch. All images must have the same resolution.')
        self._resolution = resolution

        # Load the image file and append it to the images attribute.
        with Image.open(filepath) as img:
            grayscale = img.convert('L')
            self._images.append(np.array(grayscale)/255.0)
        
    def upscale(self,
                magnification: float,
                order = 3):
        '''
        Upscale the full field images by a given magnification factor and interpolation order. The images are interpolated
        using the skimage.transform.resize function.
        
        Args:
            magnification (float):
                The magnification factor to upscale the images by. In theory, a floating point number can be used,
                but integer values are recommended. For example, a magnification factor of 2 will double the resolution.
            order (int):
                The order of the interpolation. Default is 3.
        '''
        # Check to see if the image has been upscaled. If so, raise a ValueError.
        if self._metadata.get('upscaled', False):
            raise ValueError('Images have already been upscaled. Interpolating further may have unexpected results.')
        
        for index, image in enumerate(self._images):
            # Calculate the new resolution.
            new_resolution = self._resolution/magnification

            # Calculate the new shape of the image.
            new_shape = (image.shape[0]*magnification, image.shape[1]*magnification)

            # Upscale the image.
            self._images[index] = resize(image, new_shape, order = order)

        # Update the resolution.
        self._resolution = new_resolution
        # Update the metadata so the user knows the image has been upscaled.
        self._metadata['upscaled'] = True
        self._metadata['magnification'] = magnification

    def segment_particles(self,
                          minimum_area = 10,
                          padding = 1):
        '''
        Segment the particles in the full field images. A otsu threshold is applied to separate the particles from the 
        background, connected regions are labeled, and bounding boxes are calculated for each particle. The Particle objects
        are stored as a list in the particles attribute.

        Args:
            minimum_area (int):
                The minimum area of a particle in pixels. Default 10. Any particle with an area less than this value will
                be discarded.
            padding (int):
                The number of pixels to pad the bounding box of the particle. Default is 1.
        Returns:
            None
        '''
        # Create the particles list.
        self._particles = []
        # Set the number of particles attribute.
        self._num_particles = 0

        for image in self._images:
            # Apply the otsu threshold, and create the binary mask.
            threshold = threshold_otsu(image)
            binary_mask = image > threshold

            # Label the connected regions.
            labeled = label(binary_mask)

            # Get the region properties.
            regions = regionprops(labeled)

            for region in regions:
                # Get the bounding box.
                bbox = region.bbox

                # Pad the bounding box.
                bbox = (bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding)

                # Skipping particles whose bounding box touches the edge of the image.
                if bbox[0] <= 0 or bbox[1] <= 0 or bbox[2] >= image.shape[0] or bbox[3] >= image.shape[1]:
                    continue

                # Skipping particles whose area is less than the minimum area.
                if region.area < minimum_area:
                    continue
                
                # Get the image of the particle.
                particle_image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
                # Get the binary mask of the particle.
                particle_mask = labeled[bbox[0]:bbox[2], bbox[1]:bbox[3]] == region.label

                # Create the Particle object.
                particle = Particle(image = particle_image,
                                    resolution = self._resolution,
                                    bbox = bbox,
                                    binary_mask = particle_mask,
                                    particle_id = self._num_particles)
            
                # Append the Particle object to the particles list.
                self._particles.append(particle)
                
                # Increment the number of particles attribute.
                self._num_particles += 1

    def skeletonize_particles(self, method = 'zhang'):
        '''
        Skeletonize the particles in particles attribute.

        Args:
            method (str):
                The method to use for skeletonization. Default is 'lee'. Options are 'lee' and 'zhang'. See the
                documentation for skimage.morphology.skeletonize for more information.
        Returns:
            None
        '''
        for particle in self._particles:
            particle.skeletonize_particle(method = method)

    def classify_particles(self):
        '''
        Classify the particles in the particles attribute.

        Args:
            None
        Returns:
            None
        '''
        for particle in self._particles:
            particle.classify()

    def interpolate_skeletons(self,
                              step_size: float,
                              k: int = 3,
                              s: float = 0.5):
        '''
        Interpolate the particle skeleton for each particle in the particles attribute. This is necessary for calculating
        the persistence length with subpixel accuracy. Each path is interpolated using a spline of order k with a smoothing
        factor s.

        Args:
            step_size (float):
                The step size in nanometers along the contour length of the skeleton.
            k (int):
                The degree of the spline. Default is 3.
            s (float):
                The smoothing factor. Default is 0.5.
        Returns:
            None
        '''
        self._contour_lengths = []
        for particle in self._particles:
            particle.interpolate_skeleton(step_size, k=k, s=s)
            self._contour_lengths.extend(particle.contour_lengths)
        self._contour_lengths = np.array(self._contour_lengths)

    def calc_tantan_correlations(self):
        '''
        Calculate the persistence length of the polymer particles using the Tan-Tan correlation method.

        Args:
            None
        Returns:
            None
        '''
        # Initialize the unnormalized contour length arrays and correlation arrays
        contour_samplings = []
        tantan_correlations = []

        # Calculate the Tan-Tan correlation for each particle.
        for particle in self._particles:
            particle.calc_tantan_correlation()
            contour_samplings.extend(particle.contour_samplings)
            tantan_correlations.extend(particle.tantan_correlations)

        # Find the maximum size of the contour arrays.
        max_size = max([len(contour) for contour in contour_samplings])
        # Pad the contour and correlation arrays so each array is the same size.
        padded_contours = np.array([
            np.pad(contour, (0, max_size - len(contour)), 'constant', constant_values = np.nan) for contour in contour_samplings])
        padded_correlations = np.array([
            np.pad(corr, (0, max_size - len(corr)), 'constant', constant_values = np.nan) for corr in tantan_correlations])

        # Get the contour array containing no nan values. This is the real space lag array.
        self._contour_sampling = padded_contours[[~np.isnan(lengths).any() for lengths in padded_contours]][0]

        # Calculate the mean correlation for each lag.
        self._mean_tantan_correlation = np.abs(np.nanmean(padded_correlations, axis = 0))

        # Calculate the standard deviation for error bars.
        self._tantan_std = np.nanstd(padded_correlations, axis=0)

        # Calculate the SEM (optional, preferred for error bars).
        sample_count = np.sum(~np.isnan(padded_correlations), axis=0)
        self._tantan_sem = self._tantan_std / np.sqrt(sample_count)

    def calc_persistence_length(self,
                                min_contour_length: float = 0,
                                max_contour_length: float = np.inf):
        '''
        Calculate the persistence length of the polymer particles using the Tan-Tan correlation method. The correlation will
        only be fit between the minimum and maximum contour lengths.
        
        Args:
            min_contour_length (float):
                The minimum contour length to fit the exponential decay to. Default is 0.
            max_contour_length (float):
                The maximum contour length to fit the exponential decay to. Default is np.inf.
        Returns:
            float:
                The persistence length of the polymer particles.
        '''        
        # Get the contour array containing no nan values.
        xvals = self._contour_sampling

        # Get the mask for the xvalues between the minimum and maximum contour lengths.
        inbetween_mask = (xvals >= min_contour_length) * (xvals <= max_contour_length)

        # Set the minimum and maximum contour lengths attributes for usage in the plotting methods.
        self._min_contour_length = min_contour_length
        self._max_contour_length = max_contour_length

        # Filter the xvals array to between the minimum and maximum contour lengths.
        xvals = xvals[inbetween_mask]

        # Filter the mean_correlations array to the same size as xvals.
        yvals = self._mean_tantan_correlation[inbetween_mask]

        # Fit the exponential decay to the data.
        popt, pcov = curve_fit(self.__exp_decay, xvals, yvals, p0 = 10)

        # Set the persistence length attribute.
        self._pl = popt[0]
        # Set the persistence covariance attribute.
        self._plcov = pcov

        return popt[0] * self._resolution, pcov

    def get_filtered_particles(self,
                               filter_str: str) -> list:
        '''
        Returns a list of particles that match a classification string. 

        Args:
            filter_str (str):
                The classification string to filter the particles by. Options are 'Linear', 'Branched', 'Loop', and
                'Unknown'. Note: The classification strings are case sensitive.
        Returns:
            list:
                List of Particle objects that match the classification string.
        '''
        return [particle for particle in self._particles if particle.classification == filter_str]

    def plot_image(self,
                   index: int = 0,
                   cmap = 'Greys_r'):
        '''
        Plot a full field image of the polymer data.

        Args:
            index (int):
                The index of the image to plot.
            cmap (str):
                The colormap to use for the image. Default is 'Greys_r'.
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        
        # Plot the image.
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(self._images[index], cmap=cmap)
        ax.set_title(f'Polymer Field {index}')
        return fig, ax
    
    def plot_particle(self,
                      index: int = 0):
        '''
        Plot a single particle in the particles attribute.

        Args:
            index (int):
                The index of the particle to plot.
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        return self._particles[index].plot_particle()
    
    def plot_skeleton(self,
                      index: int = 0):
        '''
        Plot the skeleton of a single particle in the particles attribute.

        Args:
            index (int):
                The index of the particle to plot.
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        return self._particles[index].plot_skeleton()

    def plot_contour_distribution(self,
                                  **kwargs):
        '''
        Plot the distribution of contour lengths for all particles. Uses Gaussian KDE to return a smooth distribution.

        Args:
            None
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''

        # Handle Keyword Arguments for plotting.
        figsize = kwargs.pop('figsize', (5, 5))
        dist_color = kwargs.pop('dist_color', 'Blue')
        fill_color = kwargs.pop('fill_color', 'LightBlue')
        lw = kwargs.pop('lw', 2)

        # If any keyword arguments are left, they aren't handled. Raise a ValueError.
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs.keys()}')

        # Create a distribution of all the polymer branch lengths.
        xvals = np.linspace(0, self._contour_lengths.max(), 1000)
        kde = gaussian_kde(self._contour_lengths)

        # Plot the distribution.
        fig, ax = plt.subplots(figsize = figsize)

        if self._min_contour_length != 0 or self._max_contour_length != np.inf:

            inbetween_mask = (xvals >= self._min_contour_length) * (xvals <= self._max_contour_length)
            less_mask = xvals <= self._min_contour_length
            greater_mask = xvals >= self._max_contour_length

            # Plot the distribution between the minimum and maximum contour lengths.
            ax.plot(xvals[inbetween_mask], kde(xvals[inbetween_mask]), color = dist_color, lw = lw, label = 'Fitting Distribution')
            # Fill the distribution between the minimum and maximum contour lengths.
            ax.fill_between(xvals[inbetween_mask], kde(xvals[inbetween_mask]), color = fill_color)
            # Plot the distribution outside the minimum and maximum contour lengths.
            ax.plot(xvals[less_mask], kde(xvals[less_mask]), color = 'Gray', alpha = 0.5, label = 'Excluded Distribution')
            ax.plot(xvals[greater_mask], kde(xvals[greater_mask]), color = 'Gray', alpha = 0.5)

            # Draw the vertical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_contour_length, color = dist_color, linestyle = '--')
            ax.axvline(self._max_contour_length, color = dist_color, linestyle = '--')

        else:
            # Plot the distribution.
            ax.plot(xvals, kde(xvals), color = dist_color, lw = lw, label = 'Contour Length Distribution')
            # Fill the distribution.
            ax.fill_between(xvals, kde(xvals), color = fill_color)

        # Set the title, xlabel, and ylabel.
        ax.set_title('Contour Length Distribution')
        ax.set_xlabel(f'Contour Length ({self._resolution:.1f} nm)')
        ax.set_ylabel('Density')
        # Add a legend.
        ax.legend()

        # Return the figure and axis objects.
        return fig, ax
    
    def plot_mean_tantan_correlation(self,
                                     error_bars: bool = False,
                                     **kwargs):
        '''
        Plot the Tan-Tan correlation for all particles.

        Args:
            error_bars (bool):
                Whether or not to plot error bars. Default is False.
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        # If error bars are set, plot the mean Tan-Tan correlation with error bars. Otherwise, plot the mean Tan-Tan
        # correlation without error bars.
        if error_bars:
            error = self._tantan_sem
        else:
            error = np.zeros_like(self._mean_tantan_correlation)

        # Handle Keyword Arguments for plotting.
        figsize = kwargs.pop('figsize', (8, 8))
        lw = kwargs.pop('lw', 2)
        corr_color = kwargs.pop('corr_color', 'Blue')
        error_color = kwargs.pop('error_color', 'LightBlue')
        fit_color = kwargs.pop('fit_color', 'Orange')

        # If any keyword arguments are left, they aren't handled. Raise a ValueError.
        if kwargs:
            raise ValueError(f'Invalid keyword arguments: {kwargs.keys()}')
        
        # Plot the Tan-Tan correlation.
        fig, ax = plt.subplots(figsize=figsize)


        # If the minimum and maximum contour lengths are set, plot a vertical line at the minimum and maximum contour
        # lengths.
        if self._min_contour_length != 0 or self._max_contour_length != np.inf:
            
            inbetween_mask = (self._contour_sampling >= self._min_contour_length) * (self._contour_sampling <= self._max_contour_length)

            # Plot the mean Tan-Tan correlation with error bars.
            ax.errorbar(self._contour_sampling[inbetween_mask], self._mean_tantan_correlation[inbetween_mask],
                        yerr = error[inbetween_mask], fmt = '.', color = corr_color, label = 'Fitting Abs Mean Correlation', ecolor = error_color)
            ax.errorbar(self._contour_sampling[~inbetween_mask], self._mean_tantan_correlation[~inbetween_mask],
                        yerr = error[~inbetween_mask], fmt = '.', color = 'Gray', alpha = 0.5, label = 'Excluded Abs Mean Correlation', ecolor = 'LightGray')
            
            # Draw the verical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_contour_length, color = corr_color, linestyle = '--')
            ax.axvline(self._max_contour_length, color = corr_color, linestyle = '--')
        else:
            # Plot the mean Tan-Tan correlation with error bars.
            ax.errorbar(self._contour_sampling, self._mean_tantan_correlation,
                        yerr = error, fmt = '.', color = corr_color, label = 'Abs Mean Correlation', ecolor = error_color)

        if self._pl != 0:
            # Plot the fitted exponential decay.
            ax.plot(self._contour_sampling, self.__exp_decay(self._contour_sampling, self._pl),
                    color=fit_color, lw=lw, label = f'Exponential Decay Fit (PL = {self._pl*self._resolution:.1f} nm)')
          
        ax.set_title('Mean Tan-Tan Correlation')
        ax.set_xlabel(f'Contour Length ({self._resolution:.1f} nm)')
        ax.set_ylabel('Correlation')
        ax.set_ylim([0,1.05])

        ax.legend()

        return fig, ax

    def print_summary(self):
        '''
        Print a summary of the polymer data.

        Args:
            None
        Returns:
            None
        '''

        print('Polymer Data Summary')
        print('---------------------')
        print('Image Stats:')
        print(f'Number of Images:\t\t{len(self._images)}')
        print(f'Interpolated:\t\t\t{self._metadata.get("upscaled", False)}')
        print(f'Resolution:\t\t\t{self._resolution:.1f} nm/pixel')
        print('---------------------')
        print('Particle Stats:')
        print(f'Number of Particles:\t\t{self._num_particles}')
        print(f'Linear Particles:\t\t{len(self.get_filtered_particles("Linear"))}')
        print(f'Branched Particles:\t\t{len(self.get_filtered_particles("Branched"))}')
        print(f'Branched-Looped Particles:\t{len(self.get_filtered_particles("Branched-Looped"))}')
        print(f'Looped Particles:\t\t{len(self.get_filtered_particles("Loop"))}')
        print(f'Unknown Particles:\t\t{len(self.get_filtered_particles("Unknown"))}')
        print('---------------------')
        print('Persistence Stats:')
        print(f'Persistence Length:\t\t{self._pl * self._resolution:.1f} nm')
        print(f'Persistence Error:\t\t{np.sqrt(np.diag(self._plcov))[0] * self._resolution:.1f} nm')
        
    @property
    def images(self) -> list[np.ndarray]:
        '''
        The list of full field images of the polymer data.
        '''
        return self._images
    
    @property
    def resolution(self) -> float:
        '''
        The resolution of the image in nanometers per pixel.
        '''
        return self._resolution
    
    @property
    def particles(self) -> list[Particle]:
        '''
        List of Particle objects.
        '''
        return self._particles
    
    @property
    def metadata(self) -> Dict[str, Any]:
        '''
        The metadata associated with the polymer image. Key-value pairs of metadata.
        '''
        return self._metadata
    
    @property
    def num_particles(self) -> int:
        '''
        The number of particles in the particles attribute.
        '''
        return self._num_particles

    @property
    def contour_lengths(self) -> list:
        '''
        The contour lengths of all paths in all particles.
        '''
        return self._contour_lengths

    @property
    def contour_sampling(self) -> np.ndarray:
        '''
        The contour sampling used for all paths in all particles. It ranges from 0 to the maximum contour length in steps of
        step_size as set in the interpolate_particles method.
        '''
        return self._contour_sampling

    @property
    def mean_tantan_correlation(self) -> np.ndarray:
        '''
        The mean Tan-Tan correlation of all particles. Calculated by taking the mean of the Tan-Tan correlation for each
        path in each particle. 
        '''
        return self._mean_tantan_correlation


    @property
    def pl(self) -> float:
        '''
        The persistence length of the polymer particles in nanometers.
        '''
        return self._pl

    @property
    def plcov(self) -> float:
        '''
        The covariance of the persistence length of the polymer particles in nanometers.
        '''
        return self._plcov

    @property
    def percentile_threshold(self) -> float:
        '''
        The threshold of the nth percentile of the contour length distribution. This is used to fit the exponential decay
        during the calculation of the persistence length. 
        '''
        return self._percentile_threshold