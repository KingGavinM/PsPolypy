from typing import Any, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from scipy.interpolate import splprep, splev
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit

from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize

import skan
import networkx as nx
import lmfit

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
                 id: int = None) -> None:
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
        self._id = id

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
            self._classification = 'Looped'

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

    def calc_tantan_correlation(self) -> None:
        '''
        Calculate the tangent-tangent correlation for each path in the particle. 

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

    def plot_particle(self,
                      ax: plt.Axes = None,
                      **kwargs) -> plt.Axes:
        '''
        Plot the particle image.

        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # Create the ax object if it is not set.
        ax = ax or plt.gca()
        # Plot the image.
        ax.imshow(self._image, **kwargs)
        # Return the fig and ax.
        return ax
    
    def plot_skeleton(self,
                      ax: plt.Axes = None,
                      **kwargs) -> plt.Axes:
        '''
        Plot the particle skeleton.
        
        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            **kwargs:
                Keyword arguments to pass to matplotlib.pyplot.imshow.
        Returns:
            fig, ax:
                The matplotlib figure and axis objects.
        '''
        # Check to see if the ske
        if self._skeleton is None:
            ValueError('Skeleton attribute is not set. Skeletonize the particle before plotting.')

        # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Plot the skeleton
        ax.imshow(self.skeleton.skeleton_image, **kwargs)
        
        # Return the ax.
        return ax
    
    def plot_interpolated_skeleton(self,
                                   ax: plt.Axes = None,
                                   **kwargs) -> plt.Axes:
        '''
        Plot the interpolated skeleton of the particle.

        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            **kwargs:
                Keyword arguments to pass to matplotlib.pyplot.plot.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Check to see if the interpolated skeleton coordinates are set. If not, raise a ValueError.
        if self._interp_skeleton_coordinates is None:
            raise ValueError('Interpolated skeleton coordinates attribute is not set. Interpolate the skeleton before plotting.')
        
        # Plot the interpolated skeletons.
        for (splinex, spliney) in self._interp_skeleton_coordinates:
            ax.plot(spliney, splinex, **kwargs)
        
        # Return the ax.
        return ax
    
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
    def id(self) -> int:
        '''
        The unique id of the particle.
        '''
        return self._id
    
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
                 **metadata: Any) -> None:
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
        self._min_fitting_length = 0

        # Set the maximum contour length attribute.
        self._max_fitting_length = np.inf

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

    def __exponential_model(self, x, lp):
        '''
        Exponential decay model for curve fitting.

        Args:
            x (float):
                The x value.
            lp (float):
                The persistence length.
        Returns:
            float:
                The exponential decay value.
        '''
        return np.exp(-x/(2*lp))
    
    ########################
    ##### Main Methods #####
    ########################

    def add_image(self,
                  filepath: str,
                  resolution: float) -> None:
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
                order = 3) -> None:
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
                          padding = 1) -> None:
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
                                    id = self._num_particles)
            
                # Append the Particle object to the particles list.
                self._particles.append(particle)
                
                # Increment the number of particles attribute.
                self._num_particles += 1

    def skeletonize_particles(self, method = 'zhang') -> None:
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

    def classify_particles(self) -> None:
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
                              s: float = 0.5) -> None:
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

    def calc_tantan_correlations(self,
                                 included_classifications: list[str] = ['Linear', 'Branched', 'Loop', 'Branched-Looped']) -> None:
        '''
        Calculate the Tan-Tan correlation a set of particles. The correlation is calculated for each
        path in each particle. The correlation is then averaged over all particles and the absolute value is taken. The
        standard deviation and standard error of the mean are also calculated.

        Optionally, the user can specify which classifications to include in the correlation calculation. By default, all
        classifications are included.

        Args:
            included_classifications (list):
                The list of classifications to include in the correlation calculation. 
                Default is ['Linear', 'Branched', 'Loop', 'Branched-Looped'].
        Returns:
            None
        '''
        # Initialize the unnormalized contour length arrays and correlation arrays
        contour_samplings = []
        tantan_correlations = []

        # Filter the particles by the included classifications.
        for classification in included_classifications:
            particles = self.get_filtered_particles(classification)

            # Calculate the Tan-Tan correlation for each particle matching the classification.
            for particle in particles:
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
        self._mean_tantan_std = np.nanstd(padded_correlations, axis=0)
        # Calculate the SEM (optional, preferred for error bars).
        sample_count = np.sum(~np.isnan(padded_correlations), axis=0)
        self._mean_tantan_sem = self._mean_tantan_std / np.sqrt(sample_count)
    
    def calc_tantan_lp(self,
                       lp_init = 10,
                       min_fitting_length: float = 0,
                       max_fitting_length: float = np.inf) -> None:
        '''
        Calculate the persistence length of the polymer particles using the Tan-Tan correlation method. The correlation will
        only be fit between the minimum and maximum contour lengths. This method uses the lmfit package for curve fitting.

        The persistence length is calculated using the formula:
        <cos(theta)> = exp(-L/Lp)
        where L is the contour length of the skeleton, and Lp is the persistence length.

        Args:
            min_fitting_length (float):
                The minimum contour length to fit the exponential decay to. Default is 0.
            max_fitting_length (float):
                The maximum contour length to fit the exponential decay to. Default is np.inf.
        Returns:
            None
        '''
        # Get the mask for the xvalues between the minimum and maximum contour lengths.
        inbetween_mask = (self._contour_sampling >= min_fitting_length) * (self._contour_sampling <= max_fitting_length)

        # Set the minimum and maximum contour lengths attributes for usage in the plotting methods.
        self._min_fitting_length = min_fitting_length
        self._max_fitting_length = max_fitting_length

        # Filter the xvals array to between the minimum and maximum contour lengths.
        xvals = self._contour_sampling[inbetween_mask]

        # Filter the mean_correlations array to the same size as xvals.
        yvals = self._mean_tantan_correlation[inbetween_mask]

        # Filter the mean_tantan_sem array to the same size as xvals and invert it to get the weights.
        weights = 1 / self._mean_tantan_sem[inbetween_mask]

        # Create a Model object
        model = lmfit.Model(self.__exponential_model)

        # Create a Parameters object
        params = model.make_params(lp = lp_init)

        # Fit the model to the data.
        result = model.fit(yvals, params, x = xvals, weights = weights)

        # Set the tantan_fit_result attribute.
        self._tantan_fit_result = result

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
                   ax: plt.Axes = None,
                   **kwargs) -> plt.Axes:
        '''
        Plot a full field image of the polymer data.

        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            index (int):
                The index of the image to plot.
            **kwargs:
                Keyword arguments to pass to matplotlib.pyplot.imshow.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # If the axis object is not set, get the current axis.
        ax = ax or plt.gca()

        # Create the fig and ax.
        ax.imshow(self._images[index], **kwargs)
        return ax
    
    def plot_particle(self,
                      index: int,
                      ax: plt.Axes = None,
                      **kwargs) -> plt.Axes:
        '''
        Plot a single particle in the particles attribute.

        Args:
            index (int):
                The index of the particle to plot.
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        return self._particles[index].plot_particle(ax = ax, **kwargs)
    
    def plot_skeleton(self,
                      index: int,
                      ax: plt.Axes = None,
                      **kwargs) -> plt.Axes:
        '''
        Plot the skeleton of a single particle in the particles attribute.

        Args:
            index (int):
                The index of the particle to plot.
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            **kwargs:
                Keyword arguments to pass to matplotlib.pyplot.imshow.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        return self._particles[index].plot_skeleton(ax = ax, **kwargs)
    
    def plot_interpolated_skeleton(self,
                                   index: int,
                                   ax: plt.Axes = None,
                                   **kwargs) -> plt.Axes:
        '''
        Plot the interpolated skeleton of a single particle in the particles attribute.

        Args:
            index (int):
                The index of the particle to plot.
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            **kwargs:
                Keyword arguments to pass to matplotlib.pyplot.plot.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        return self._particles[index].plot_interpolated_skeleton(ax = ax, **kwargs)
    
    def plot_contour_distribution(self,
                                  n_points: int = 100,
                                  ax: plt.Axes = None,
                                  inc_dist_kwargs: dict = None,
                                  inc_fill_kwargs: dict = None,
                                  exc_dist_kwargs: dict = None,
                                  exc_fill_kwargs: dict = None,
                                  vline_kwargs: dict = None) -> plt.Axes:
        '''
        Plot the distribution of contour lengths for all particles. Uses Gaussian KDE to return a smooth distribution.

        Args:
            n_points (int):
                The number of points to use for the Gaussian KDE. Default is 100.
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            inc_dist_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the included distribution. (Between the minimum and
                maximum contour lengths used for fitting) If the minimum and maximum contour lenths are 0 and np.inf, these
                kwargs will be used for the entire distribution.
                Default is None
            inc_fill_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.fill_between for the indcluded distribution.
                Default is None.
            exc_dist_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the excluded distribution. (Outside the minimum and
                maximum contour lengths used for fitting) 
                Default is None.
            exc_fill_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.fill_between for the excluded distribution.
                Default is None
            vline_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.axvline for the vertical lines at the minimum and maximum
                contour lengths.
                Default is None
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Handle the default kwargs if they are none.
        if inc_dist_kwargs is None:
            inc_dist_kwargs = {}
        if inc_fill_kwargs is None:
            inc_fill_kwargs = {}
        if exc_dist_kwargs is None:
            exc_dist_kwargs = {}
        if exc_fill_kwargs is None:
            exc_fill_kwargs = {}
        if vline_kwargs is None:
            vline_kwargs = {}

        # Create a distribution of all the polymer branch lengths.
        xvals = np.linspace(0, self._contour_lengths.max(), n_points)
        kde = gaussian_kde(self._contour_lengths)

        # If the minimum and maximum contour lengths are set, only color the region between the minimum and maximum, plot
        # the excluded regions in gray and plot vertical lines at the minimum and maximum contour lengths.
        if self._min_fitting_length != 0 or self._max_fitting_length != np.inf:
            
            # Get the mask for the xvalues less than, in between, and greather than the min and max contour lengths.
            inbetween_mask = (xvals >= self._min_fitting_length) * (xvals <= self._max_fitting_length)
            less_mask = xvals <= self._min_fitting_length
            greater_mask = xvals >= self._max_fitting_length

            # Plot the distribution between the minimum and maximum contour lengths.
            ax.plot(xvals[inbetween_mask], kde(xvals[inbetween_mask]), **inc_dist_kwargs)
            # Fill the distribution between the minimum and maximum contour lengths.
            ax.fill_between(xvals[inbetween_mask], kde(xvals[inbetween_mask]), **inc_fill_kwargs)

            # Plot the distribution outside the minimum and maximum contour lengths.
            ax.plot(xvals[less_mask], kde(xvals[less_mask]), **exc_dist_kwargs)
            ax.fill_between(xvals[less_mask], kde(xvals[less_mask]), **exc_fill_kwargs)

            # Create a copy of the dist kwargs without the label so it isn't shown in the legend twice.
            exc_dist_kwargs = exc_dist_kwargs.copy()
            exc_dist_kwargs.pop('label', None)
            ax.plot(xvals[greater_mask], kde(xvals[greater_mask]), **exc_dist_kwargs)
            ax.fill_between(xvals[greater_mask], kde(xvals[greater_mask]), **exc_fill_kwargs)

            # Draw the vertical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_fitting_length, **vline_kwargs)
            ax.axvline(self._max_fitting_length, **vline_kwargs)

        else:
            # Plot the distribution.
            ax.plot(xvals, kde(xvals), **inc_dist_kwargs)
            # Fill the distribution.
            ax.fill_between(xvals, kde(xvals), **inc_fill_kwargs)
        
        # Return the ax.
        return ax
    
    def plot_mean_tantan_correlation(self,
                                     error_bars: bool = False,
                                     ax: plt.Axes = None,
                                     inc_kwargs: dict = None,
                                     exc_kwargs: dict = None,
                                     vline_kwargs: dict = None) -> plt.Axes:
        '''
        Plot the Tan-Tan correlation for all particles.

        Args:
            error_bars (bool):
                Whether or not to plot error bars. Default is False.
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            inc_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.errorbar for the included data points. (Between the minimum
                and maximum contour lengths used for fitting) If the minimum and maximum contour lenths are 0 and np.inf,
                these kwargs will be used for the entire dataset.
                Default is None.
            exc_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.errorbar for the excluded data points. (Outside the minimum
                and maximum contour lengths used for fitting) 
                Default is None.
            vline_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.axvline for the vertical lines at the minimum and maximum
                contour lengths.
                Default is None.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Handle the default kwargs if they are none.
        if inc_kwargs is None:
            inc_kwargs = {}
        if exc_kwargs is None:
            exc_kwargs = {}
        if vline_kwargs is None:
            vline_kwargs = {}

        # If the error bars are set, the error is the standard error of the mean. Otherwise, the error is 0.
        if error_bars:
            error = self._mean_tantan_sem
        else:
            error = np.zeros_like(self._mean_tantan_correlation)

        # If the minimum and maximum contour lengths are set, only color the region between the minimum and maximum, plot
        # the excluded regions in gray and plot vertical lines at the minimum and maximum contour lengths.
        if self._min_fitting_length != 0 or self._max_fitting_length != np.inf:
            
            # Get the mask for the xvalues in between the min and max contour lengths
            inbetween_mask = (self._contour_sampling >= self._min_fitting_length) * (self._contour_sampling <= self._max_fitting_length)

            # Plot the mean Tan-Tan correlation between the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[inbetween_mask],
                        self._mean_tantan_correlation[inbetween_mask],
                        yerr = error[inbetween_mask],
                        **inc_kwargs)
            # Plot the mean Tan-Tan correlation outside the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[~inbetween_mask],
                        self._mean_tantan_correlation[~inbetween_mask],
                        yerr = error[~inbetween_mask],
                        **exc_kwargs)
            
            # Draw the verical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_fitting_length, **vline_kwargs)
            ax.axvline(self._max_fitting_length, **vline_kwargs)

        else:
            # Plot the mean Tan-Tan correlation with error bars.
            ax.errorbar(self._contour_sampling,
                        self._mean_tantan_correlation,
                        yerr = error,
                        **inc_kwargs)
        
        return ax

    def plot_mean_tantan_correlation_fit(self,
                                         ax: plt.Axes = None,
                                         show_init: bool = False,
                                         fit_kwargs: dict = None,
                                         init_kwargs: dict = None) -> plt.Axes:
        '''
        Plot the fitted exponential decay of the mean Tan-Tan correlation.

        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            fit_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the fitted exponential decay.
                Default is None.
            init_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the initial guess of the exponential decay.
                Only used if show_init is True.
                Default is None. 
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''
        # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Handle the default kwargs if they are none.
        if fit_kwargs is None:
            fit_kwargs = {}
        if init_kwargs is None:
            init_kwargs = {}

        # Plot the fitted exponential model.
        ax.plot(self._contour_sampling,
                self.__exponential_model(self._contour_sampling, self._tantan_fit_result.params['lp'].value),
                **fit_kwargs)
        
        # If show_init is true, also plot the initial guess.
        if show_init:
            ax.plot(self._contour_sampling,
                    self.__exponential_model(self._contour_sampling, self._tantan_fit_result.params['lp'].init_value),
                    **init_kwargs)
        
        return ax

    def print_summary(self) -> None:
        '''
        Print a summary of the polymer data.

        Args:
            None
        Returns:
            None
        '''

        print('Polymer Data Summary')
        print('-'*57)
        print('Image Stats:')
        print(f'Number of Images:\t\t\t{len(self._images)}')
        print(f'Interpolated:\t\t\t\t{self._metadata.get("upscaled", False)}')
        print(f'Resolution:\t\t\t\t{self._resolution:.1f} nm/pixel')
        print('-'*57)
        print('Particle Stats:')
        print(f'Number of Particles:\t\t\t{self._num_particles}')
        print(f'Linear Particles:\t\t\t{len(self.get_filtered_particles("Linear"))}')
        print(f'Branched Particles:\t\t\t{len(self.get_filtered_particles("Branched"))}')
        print(f'Branched-Looped Particles:\t\t{len(self.get_filtered_particles("Branched-Looped"))}')
        print(f'Looped Particles:\t\t\t{len(self.get_filtered_particles("Looped"))}')
        print(f'Unknown Particles:\t\t\t{len(self.get_filtered_particles("Unknown"))}')
        print('-'*57)
        print('Persistence Length Stats:')
        print(f'Tan-Tan Correlation lp:\t\t\t{self._tantan_fit_result.params["lp"].value * self._resolution:.1f} +/- {self._tantan_fit_result.params["lp"].stderr * self._resolution:.1f} nm')
        print(f'Tan-Tan Correlation Reduced Chi^2:\t{self._tantan_fit_result.redchi:.2f}')
        
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
    def tantan_fit_result(self) -> lmfit.model.ModelResult:
        '''
        The lmfit model result of the Tan-Tan correlation fit.
        '''
        return self._tantan_fit_result