from typing import Any, Tuple, Dict
import copy

import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev
from scipy import integrate, interpolate
from scipy.stats import gaussian_kde

from skimage import io
from skimage.util import img_as_float
from skimage.color import rgb2gray, rgba2rgb
from skimage.transform import resize
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize


import skan
import networkx as nx
import lmfit
from lmfit import Minimizer, Parameters

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

        # Set the displacements attribute.
        self._displacements = None

        # Set the Tan-Tan correlation attribute.
        self._tantan_correlations = None

    def skeletonize_particle(self,
                             method: str = 'zhang') -> Tuple[skan.Skeleton, 'pandas.DataFrame']: # type: ignore
        '''
        Skeletonize the particle's binary mask and set the skeleton attribute.

        Args:
            method (str):
                The method to use for skeletonization. Default is 'lee'. Options are 'lee' and 'zhang'. See the
                documentation for skimage.morphology.skeletonize for more information.
        Returns:
            None
        '''
        # Skeletonize the binary mask with skimage.
        skeletonized = skeletonize(self._binary_mask, method = method)   

        # Create the skan skeleton object.
        skan_skeleton = skan.Skeleton(skeletonized, source_image = self._image)

        # Set the skeleton attribute.
        self._skeleton = skan_skeleton

        # Create the skeleton summary.
        skeleton_summary = skan.summarize(skan_skeleton, separator = '_')
        # Set the skeleton summary attribute.
        self._skeleton_summary = skeleton_summary

        # Return the skan skeleton ans skeleton summary.
        return skan_skeleton, skeleton_summary

    def classify(self) -> str:
        '''
        Classify the particle as Linear, Branched, Looped, Branched-Looped, or Unknown based on the skeleton summary.
        
        Args:
            None
        Returns:
            classification (str):
                The classification of the particle.
        '''
        # Check to see if the skeleton summary is set. If not, raise a ValueError.
        if self._skeleton_summary is None:
            raise ValueError('Skeleton summary attribute is not set. Skeletonize the particle before classifying.')
        
        # Get the unique branch types from the skeleton summary.
        unique_branch_types = self._skeleton_summary['branch_type'].unique()

        # Check if the particle is classified as linear:
        if np.all(unique_branch_types == 0):
            classification = 'Linear'

        # Check if the particle is classified as branched-loop:
        elif 1 in unique_branch_types:
            # The particle is some form of branched, now check if it contains any cycles in the graph representation.
            graph = skan.csr.skeleton_to_nx(self._skeleton)
            loops = list(nx.simple_cycles(graph))
            
            # If any loops are found, classify the particle as branched-loop.
            if len(loops) > 0:
                classification = 'Branched-Looped'
            else:
                classification = 'Branched'

        # Check if the particle is classified as looped:
        elif np.all(unique_branch_types == 3):
            classification = 'Looped'

        # If the particle does not fit any of the above classifications, set the classification to unknown.
        else:
            classification = 'Unknown'

        # Set the classification attribute.
        self._classification = classification

        # Return the classification.
        return classification

    def interpolate_skeleton(self,
                             step_size: float,
                             k: int = 3,
                             s: float = 0.5) -> Tuple[list[np.ndarray], list[float], list[np.ndarray], list[np.ndarray]]:
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
            contour_samplings, contour_lengths, interp_skeleton_coordinates, interp_skeleton_derivatives (Tuple):
                Tuple containing the contour samplings, contour lengths, interpolated skeleton coordinates, and interpolated
                skeleton derivatives for each path in the particle.
        '''
        # Check to see if the skeleton is set. If not, raise a ValueError.
        if self._skeleton is None:
            raise ValueError('Skeleton attribute is not set. Skeletonize the particle before interpolating.')

        # Initialize lists to store interpolated values for each path.
        contour_samplings = []
        contour_lengths = []
        interp_skeleton_coordinates = []
        interp_skeleton_derivatives = []

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
            contour_lengths.append(contour_length)

            # Set the actual contour coordinates for this path.
            contour_actual = np.arange(0, contour_length, step_size)

            # Set the normalized contour coordinates for this path.
            contour_normalized = contour_actual / contour_length

            # Append the true contour sampling for this path.
            contour_samplings.append(contour_actual)

            # Append the interpolated skeleton coordinates for this path.
            interp_skeleton_coordinates.append(splev(contour_normalized, tck))

            # Append the interpolated derivative for this path.
            interp_skeleton_derivatives.append(splev(contour_normalized, tck, der=1))

        # Set the attributes
        self._contour_samplings = contour_samplings
        self._contour_lengths = contour_lengths
        self._interp_skeleton_coordinates = interp_skeleton_coordinates
        self._interp_skeleton_derivatives = interp_skeleton_derivatives
        
        # Return the interpolated values.
        return contour_samplings, contour_lengths, interp_skeleton_coordinates, interp_skeleton_derivatives

    def calc_displacements(self) -> list[np.ndarray]:
        '''
        Calculate the displacment along the contour for each path in the particle.

        Args:
            None
        Returns:
            (list):
                List of displacements for each path in the particle.
        '''
        # Check to see if the interpolated skeleton coordinates are set. If not, raise a ValueError.
        if self._interp_skeleton_coordinates is None:
            raise ValueError('Interpolated skeleton coordinates attribute is not set. Interpolate the skeleton before calculating the persistence length.')
        
        # Initialize a list to store the displacements for each path.
        displacements = []
        displacement_matrices = []

        # Loop over each path's interpolated skeleton coordinates.
        for coords in self._interp_skeleton_coordinates:
            coords = np.array(coords)

            N = len(coords.T)

            # Compute the displacement matrix, and append it to the list.
            diff = coords[:, :, None] - coords[:, None, :]
            sq_diff = np.sum(diff**2, axis=0)
            disp_mat = np.sqrt(sq_diff)
            displacement_matrices.append(disp_mat)

            # The displacments for each subpath are the upper triangle of the displacement matrix.
            disp_values = [disp_mat[i, i:] for i in range(N)]
            
            # Append the displacements for this path to the list.
            displacements.append(disp_values)
        
        # Set the displacement_matrices attribute.
        self._displacement_matrices = displacement_matrices

        # Return the displacements.
        return displacements

    def calc_tantan_correlation(self) -> list[np.ndarray]:
        '''
        Calculate the tangent-tangent correlation for each path in the particle and set the tantan_correlations attribute.

        Args:
            None
        Returns:
            (list):
                List of Tan-Tan correlations for each path in the particle
        '''
        # Check to see if the interpolated skeleton derivatives are set. If not, raise a ValueError.
        if self._interp_skeleton_derivatives is None:
            raise ValueError('Interpolated skeleton derivatives attribute is not set. Interpolate the skeleton before calculating the persistence length.')

        # Initialize a list to store Tan-Tan correlations for each path.
        tantan_correlations = []
        dotproduct_matrices = []

        # Loop over each path's interpolated skeleton derivative.
        for  derivative in self._interp_skeleton_derivatives:
            # Calculate the tangent vectors for the current path.
            tangents = derivative / np.linalg.norm(derivative, axis=0, keepdims=True)

            # Check to make sure the unit tangent vectors do not have sign flips.
            # Compute the dot product of consecutive tangent vectors
            dot_products = np.sum(tangents[:, :-1] * tangents[:, 1:], axis=0)
            # Determine where the sign flips are needed
            flips = np.ones(tangents.shape[1], dtype=float)
            flips[1:] = np.cumprod(np.sign(dot_products) + (dot_products == 0))
            # Apply the flips to the tangent vectors
            tangents = tangents * flips

            # Get the number of tangent vectors.
            _, N = tangents.shape
            
            # Computer the dot product matrix and append it to the list
            dot_product_matrix = np.dot(tangents.T, tangents)
            dotproduct_matrices.append(dot_product_matrix)

            # The correlations for each subpath are the upper triangle of the dot product matrix.
            corr = [dot_product_matrix[i,i:] for i in range(N)]

            # Append the correlations for this path to the list.
            tantan_correlations.append(corr)

        # Set the dotproduct_matrices attribute.
        self._dotproduct_matrices = dotproduct_matrices

        # Return the Tan-Tan correlations.
        return tantan_correlations

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
    def displacements(self) -> list[np.ndarray]:
        '''
        The displacements of each path in the particle.
        '''
        return self._displacements
    
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
        self._images = copy.deepcopy(images) if images is not None else []

        # Set the metadata attribute.
        self._metadata = copy.deepcopy(metadata) if metadata is not None else {}

        # Set the resolution attribute.
        self._resolution = resolution
        # Set the base resolution metadata.
        self._metadata['base_resolution'] = resolution

        # Set the particles attribute.
        self._particles = []

        # Set the number of particles attribute.
        self._num_particles = {'All': 0, 'Linear': 0, 'Branched': 0, 'Looped': 0, 'Branched-Looped': 0, 'Unknown': 0}

        # Set the included_classifications attribute.
        self._included_classifications = ['Linear', 'Branched', 'Looped', 'Branched-Looped', 'Unknown']

        # Set the contour lengths attribute.
        self._contour_lengths = None

        # Set the contour sampling attribute.
        self._contour_sampling = None

        # Set the mean squared displacements attribute.
        self._mean_squared_displacements = None

        # Set the mean Tan-Tan correlations attribute.
        self._mean_tantan_correlations = None

        # Set the minimum contour length attribute.
        self._min_fitting_length = 0

        # Set the maximum contour length attribute.
        self._max_fitting_length = np.inf

        # Set the R2 fit result attribute.
        self._R2_fit_result = None

        # Set the tantan fit result attribute.
        self._tantan_fit_result = None

    ##########################
    ##### Static Methods #####
    ##########################
    
    @staticmethod
    def __load_with_skimage(filepath: str) -> np.ndarray:
        '''
        Load an image file with skimage.io.imread.

        Args:
            filepath (str):
                The file path to the image file.
        Returns:
            np.ndarray:
                The image array.
        '''
        # Load the image file.
        image = io.imread(filepath)
        # Convert the image to grayscale if it is not already.
        image = img_as_float(image)

        # If the image has 4 channels, convert RGBA to RGB.
        if image.ndim == 3 and image.shape[-1] == 4:
            image = rgba2rgb(image)

        # Convert to grayscale if the image is not already.
        if image.ndim == 3 and image.shape[-1] == 3:
            image = rgb2gray(image)

        # Return the image.
        return image
    
    @staticmethod
    def __exponential_model(x, lp):
        '''
        Exponential decay model for curve fitting.

        Args:
            x (float):
                The x value.
            lp (float):
                The persistence length.
        Returns:
            float:
                The exponential decay model value.
        '''
        return np.exp(-x/(2*lp))

    @staticmethod
    def __R2_model(x, lp):
        '''
        End to end distance sqaured model for curve fitting.

        Args:
            x (float):
                The x value.
            lp (float):
                The persistence length.
        Returns:
            float:
                The R^2 model value.
        '''
        return 2*2*lp*x * (1 - (2*lp/(x + 1e-10)) * (1 - np.exp(-x / (2*lp))))
    
    @staticmethod
    def __calc_confidence_interval(data, npoints=1000, confidence_level=0.6826):
        '''
        Calculate the confidence interval for a given dataset using a Gaussian KDE.
        The cumulative distribution function (CDF) is approximated by taking the cumulative sum of the PDF and
        multiplying by the grid spacing. This approximates the lower and upper bounds of the confidence interval,
        but is less computationally intensive as integrating the KDE for the CDF.

        Args:
            data (np.ndarray):
                The data to calculate the confidence interval for.
            npoints (int):
                The number of points to evaluate the KDE at. Default is 1000.
            alpha (float):
                The confidence level. Default is 0.6826.
        '''

        # In the case that the data only has a single element, return it as lower and upper bounds.
        if len(data) < 2:
            return data[0], data[0]
        
        # If the data is all the same, return the value as the lower and upper bounds.
        if np.isclose(np.var(data),0):
            return data[0], data[0]
        
        # Calculate the KDE
        kde = gaussian_kde(data)

        # Get the x meshpoints for the KDE
        x = np.linspace(np.min(data), np.max(data), npoints)

        # Evaluate the PDF at these points (vectorized)
        pdf_values = kde.evaluate(x)

        # Calculate CDF using cumulative trapezoidal integration
        cdf = integrate.cumulative_trapezoid(pdf_values, x, initial=0)
        cdf /= cdf[-1]  # Normalize CDF
        
        # Create the Percent Point Function (inverse of CDF)
        ppf = interpolate.interp1d(cdf, x, kind='linear', bounds_error=False, fill_value=(x[0], x[-1]))
        
        # Calculate confidence interval
        lower_percentile = (1 - confidence_level) / 2
        upper_percentile = 1 - lower_percentile
        
        lower_bound = ppf(lower_percentile)
        upper_bound = ppf(upper_percentile)
        
        return lower_bound, upper_bound

    @staticmethod
    def __confidence_interval_minimizer(data, ci, model, initial_guess):
        '''
        Creates a minimizer for fitting a model to data using the confidence interval as the objective function.

        Args:
            data (np.ndarray):
                The data to fit the model to. Shape (2,N).
            ci (np.ndarray):
                The confidence interval for the data.
            model (callable):
                The model to fit the data to.
        '''
        # Unpack the data
        xvals, yvals = data

        # Define the objective function callable
        def objective_function(params, x, y, ci, model):
            '''
            Objective function for the minimizer. The objective function is the difference between the model
            and the confidence interval.

            Args:
                params (lmfit.Parameters):
                    The parameters to fit.
                x (np.ndarray):
                    The x values.
                y (np.ndarray):
                    The y values.
                ci (np.ndarray):
                    The confidence interval.
                model (callable):
                    The model to fit the data to.
            Returns:
                np.ndarray:
                    The objective function values.
            '''
            # Set the parameters
            lp = params['lp'].value

            # Calculate the model values
            model_values = model(x, lp)

            # Calculate the residuals
            yerr = np.abs([y - ci.T[0], ci.T[1] - y])
            residuals = np.where(model_values > y, (model_values - y) / yerr[0], (y - model_values) / yerr[1])

            return residuals
        
        # Initialize the parameters
        params = Parameters()
        params.add('lp', value=initial_guess, min=0)

        return Minimizer(objective_function, params, fcn_args=(xvals, yvals, ci, model))

    #########################
    ##### Class Methods #####
    #########################

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
            # Load the image using the skimage helper function.
            image = cls.__load_with_skimage(filepath)

            # Add the image to the images list.
            images.append(image)
            
        # Create the Polydat object.
        return cls(images = images, resolution = resolution, **metadata)
    
    ########################
    ##### Main Methods #####
    ########################

    def add_image(self,
                  filepath: str,
                  resolution: float) -> list[np.ndarray]:
        '''
        Load an image file and add it to the images attribute.

        Args:
            filepath (str):
                The file path to the image file.
            resolution (float):
                The resolution of the image in nanometers per pixel.
        Returns:
            images (list):
                The list of images.
        '''
        # Make sure the resolution of the image matches the resolution of the other images.
        if self._resolution is not None and self._resolution != resolution:
            raise ValueError('Resolution mismatch. All images must have the same resolution.')
        self._resolution = resolution

        # Load the image using the skimage helper function.
        image = self.__load_with_skimage(filepath)

        # Append the image to the images attribute.
        self._images.append(image)

        # Return the images.
        return self._images
        
    def upscale(self,
                magnification: float,
                order = 3) -> Tuple[list[np.ndarray], float]:
        '''
        Upscale the full field images by a given magnification factor and interpolation order. The images are interpolated
        using the skimage.transform.resize function.
        
        Args:
            magnification (float):
                The magnification factor to upscale the images by. In theory, a floating point number can be used,
                but integer values are recommended. For example, a magnification factor of 2 will double the resolution.
            order (int):
                The order of the interpolation. Default is 3.
        Returns:
            images, resolution (Tuple):
                Tuple containing the upscaled images and the new resolution.
        '''
        # Check to see if the image has been upscaled. If so, raise a ValueError.
        if self._metadata.get('upscaled', False):
            raise ValueError('Images have already been upscaled. Interpolating further may have unexpected results.')
        
        for index, image in enumerate(self._images[:]):
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
        self._metadata['interpolation_order'] = order
        self._metadata['upscaled_resolution'] = new_resolution

        # Return
        return self._images, self._resolution

    def segment_particles(self,
                          minimum_area = 10,
                          padding = 1) -> list[Particle]:
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
            particles (list[Particle]):
                The list of Particle objects.
        '''
        # Initialize the number of particles and particle list attributes.
        self._num_particles['All'] = 0
        self._particles = []
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

                # Skipping particles with fewer than 3 pixels in the binary mask.
                if np.sum(particle_mask) < 3:
                    continue

                # Create the Particle object.
                particle = Particle(image = particle_image,
                                    resolution = self._resolution,
                                    bbox = bbox,
                                    binary_mask = particle_mask,
                                    id = region.label-1)
            
                # Append the Particle object to the particles list.
                self._particles.append(particle)
                
                # Increment the number of particles attribute.
                self._num_particles['All'] += 1
        
        # Return the particles.
        return self._particles

    def skeletonize_particles(self, method = 'zhang') -> list[Particle]:
        '''
        Skeletonize the particles in particles attribute.

        Args:
            method (str):
                The method to use for skeletonization. Default is 'lee'. Options are 'lee' and 'zhang'. See the
                documentation for skimage.morphology.skeletonize for more information.
        Returns:
            particles (list[Particle]):
        '''
        # Loop over the particles and skeletonize them.
        for particle in self._particles[:]:
            try:
                # Attempt to skeletonize the particle.
                particle.skeletonize_particle(method = method)
                # If the particle's skeleton contains fewer than 4 pixels, remove the particle from the list.
                if np.sum(particle.skeleton.skeleton_image) < 4:
                    self._particles.remove(particle)
            except ValueError:
                # If the skeletonization fails, remove the particle from the list.
                self._particles.remove(particle)

        # Return the particles.
        return self._particles


    def classify_particles(self) -> None:
        '''
        Classify the particles in the particles attribute.

        Args:
            None
        Returns:
            particles (list[Particle]):
                The list of Particle objects.
        '''
        # Loop over the particles and classify them.
        for particle in self._particles:
            particle.classify()
            self._num_particles[particle.classification] = self._num_particles.get(particle.classification, 0) + 1

        # Return the particles.
        return self._particles



    def filter_particles(self,
                         classifications: list[str]) -> list[Particle]:
        '''
        Remove particles from the particles attribute that do not match the given classifications.

        Args:
            classifications (list):
                The list of classifications to include in the filtering.
        Returns:
            particles (list[Particle]):
                The list of Particle objects.
        '''
        # Check to see if the classifications are valid.
        for classification in classifications:
            if classification not in ['Linear', 'Branched', 'Looped', 'Branched-Looped', 'Unknown']:
                raise ValueError(f'Invalid classification: {classification}. Valid classifications are Linear, Branched, Looped, Branched-Looped, and Unknown.')

        # Set the particles attribute to only include particles with the given classifications.
        self._particles = [particle for particle in self._particles if particle.classification in classifications]

        # Update the included classifications attribute.
        self._included_classifications = classifications

        # Return the filtered particles.
        return self._particles

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
    
    def calc_displacements(self) -> None:
        '''
        Calculate the mean squared displacements for a set of particles. The displacements are calculated for each path in
        each particle. The displacements are then squared and averaged over all particles. The standard deviation and standard
        error of the mean are also calculated.

        Args:
            None
        Returns:
            None
        '''
        # Initialize the contour length and displacements array.
        contour_samplings = []
        displacements_list = []

        # Calculate the displacements and contour samplings for each particle.
        for particle in self._particles:
            displacements_list.extend(particle.calc_displacements())
            contour_samplings.extend(particle.contour_samplings)
        
        # Pad the contour array so each array is the same size.
        # Find the maximum size of the contour arrays.
        max_size = max([len(contour) for contour in contour_samplings])
        # Pad the contour and displacement arrays so each array is the same size.
        padded_contours = np.array([np.pad(contour, (0, max_size - len(contour)), 'constant', constant_values = np.nan) for contour in contour_samplings])
        self._padded_contours = padded_contours
        # Get the contour sampling of the longest
        contour_sampling = padded_contours[[~np.isnan(lengths).any() for lengths in padded_contours]][0]
        # Get the contour array containing no nan values. This is the real space lag array.
        self._contour_sampling = contour_sampling
        
        # Pad the displacements so each array is the same size.
        padded_disp= []
        for displacements in displacements_list:
            padded = np.array([np.pad(displacement, (0, max_size - len(displacement)), 'constant', constant_values = np.nan) for displacement in displacements])
            padded_disp.extend(padded)
        padded_disp = np.array(padded_disp)

        padded_disp_sq = padded_disp**2

        # Calculate the mean squared displacements for each lag.
        self._mean_squared_displacements = np.nanmean(padded_disp_sq, axis = 0)

        # Calculate the confidence interval for each lag.
        ci_values = [self.__calc_confidence_interval(padded_disp_sq[:,i][~np.isnan(padded_disp_sq[:,i])]) for i in range(contour_sampling.shape[0])]
        self._displacement_sq_confidence_intervals = np.array(ci_values)

        # Calculate the standard deviation for error bars.
        self._mean_squared_displacement_std = np.nanstd(padded_disp_sq, axis=0)
    
    def calc_tantan_correlations(self) -> None:
        '''
        Calculate the Tan-Tan correlation a set of particles. The correlation is calculated for each
        path in each particle. The correlation is then averaged over all particles and the absolute value is taken. The
        standard deviation and standard error of the mean are also calculated.

        Args:
            None
        Returns:
            None
        '''
        # Initialize the contour sampling and correlation arrays
        contour_samplings = []
        correlations_list = []

        # Calculate the Tan-Tan correlation and contour sampling for each particle.
        for particle in self.particles:
            contour_samplings.extend(particle.contour_samplings)
            correlations_list.extend(particle.calc_tantan_correlation())

        # Find the maximum size of the contour arrays.
        max_size = max([len(contour) for contour in contour_samplings])
        # Pad the contour and correlation arrays so each array is the same size.
        padded_contours = np.array([np.pad(contour, (0, max_size - len(contour)), 'constant', constant_values = np.nan) for contour in contour_samplings])
        self._padded_contours = padded_contours
        contour_sampling  = padded_contours[[~np.isnan(lengths).any() for lengths in padded_contours]][0]
        # Get the contour array containing no nan values. This is the real space lag array.
        self._contour_sampling = contour_sampling

        padded_corr = []
        for correlations in correlations_list:
            padded_correlations = np.array([np.pad(corr, (0, max_size - len(corr)), 'constant', constant_values = np.nan) for corr in correlations])
            padded_corr.extend(padded_correlations)
        padded_corr = np.array(padded_corr)

        # Calculate the mean correlation for each lag.
        mean_tantan_correlation = np.nanmean(padded_corr, axis = 0)
        
        # Calculate the standard deviation for error bars.
        std_tantan_correlation = np.nanstd(padded_corr, axis = 0)

        # Calculate the confidence interval for each lag.
        ci_values = [self.__calc_confidence_interval(padded_corr[:,i][~np.isnan(padded_corr[:,i])]) for i in range(contour_sampling.shape[0])]
        self._tantan_confidence_intervals = np.array(ci_values)
        
        # Set the mean and std Tan-Tan correlation attribute.
        self._mean_tantan_correlations = mean_tantan_correlation
        self._mean_tantan_std = std_tantan_correlation

    def calc_R2_lp(self,
                    lp_init = 10,
                    min_fitting_length: float = 0,
                    max_fitting_length: float = np.inf) -> None:
        '''
        Calculate the persistence length of the polymer particles using the end to end distance squared model. The mean
        squared displacements will only be fit between the minimum and maximum contour lengths. This method uses the lmfit
        package for curve fitting.

        The persistence length is calculated using the formula:
        <R^2> = 2*s*Lp*l*(1 - s*Lp/l*(1 - exp(-l/(s*Lp))))
        where l is the contour length of the skeleton, s is the equilibration constant (1.5 for unequilibrated, and 2 for
        equilibrated), and Lp is the persistence length.

        Args:
            lp_init (float):
                The initial guess for the persistence length. Default is 10.
            min_fitting_length (float):
                The minimum contour length to fit the exponential decay to. Default is 0.
            max_fitting_length (float):
                The maximum contour length to fit the exponential decay to. Default is np.inf.
            fit_kwargs (dict):
                Keyword arguments to pass to the lmfit Model.fit() method. Default is None.
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

        # Filter the mean_squared_displacements array to the same size as xvals.
        yvals = self._mean_squared_displacements[inbetween_mask]

        # Filter the confidence interval array to the same size as xvals.
        ci = self._displacement_sq_confidence_intervals[inbetween_mask]

        # Create a Minimizer object for the confidence interval minimizer.
        minimizer = self.__confidence_interval_minimizer(data = (xvals, yvals), ci = ci, model = self.__R2_model, initial_guess = lp_init)

        # Perform the minimization.
        result = minimizer.minimize()

        # Set the R2_fit_result attribute.
        self._R2_fit_result = result

        # Return the fit result.
        return result
    
    def calc_tantan_lp(self,
                       lp_init = 10,
                       min_fitting_length: float = 0,
                       max_fitting_length: float = np.inf) -> None:
        '''
        Calculate the persistence length of the polymer particles using the Tan-Tan correlation method. The correlation will
        only be fit between the minimum and maximum contour lengths. This method uses the lmfit package for curve fitting.

        The persistence length is calculated using the formula:
        <cos(theta)> = exp(-L/(s*Lp))
        where L is the contour length of the skeleton, s is the equilibration constant (1.5 for unequilibrated, and 2 for
        equilibrated), and Lp is the persistence length.

        Args:
            lp_init (float):
                The initial guess for the persistence length. Default is 10.
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
        yvals = self._mean_tantan_correlations[inbetween_mask]

        # Filter the confidence interval array to the same size as xvals.
        ci = self._tantan_confidence_intervals[inbetween_mask]

        # Create a Minimizer object for the confidence interval minimizer.
        minimizer = self.__confidence_interval_minimizer(data = (xvals, yvals), ci = ci, model = self.__exponential_model, initial_guess = lp_init)

        # Perform the minimization.
        result = minimizer.minimize()

        # Set the tantan_fit_result attribute.
        self._tantan_fit_result = result

        # Return the fit result.
        return result

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
            inc_dist_kwargs = {'color': 'Blue', 'lw': 2, 'label': 'Included data'}
        if inc_fill_kwargs is None:
            inc_fill_kwargs = {'color': 'LightBlue', 'alpha': 0.5}
        if exc_dist_kwargs is None:
            exc_dist_kwargs = {'color': 'Gray', 'lw': 2, 'alpha': 0.5, 'label': 'Excluded data'}
        if exc_fill_kwargs is None:
            exc_fill_kwargs = {'color': 'LightGray', 'alpha': 0.5}
        if vline_kwargs is None:
            vline_kwargs = {'color': 'Blue', 'lw': 0.7, 'dashes': [8,3]}

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
    
    def plot_subpath_contour_distribution(self,
                                          n_points: int = 100,
                                          ax: plt.Axes = None,
                                          inc_dist_kwargs: dict = None,
                                          inc_fill_kwargs: dict = None,
                                          exc_dist_kwargs: dict = None,
                                          exc_fill_kwargs: dict = None,
                                          vline_kwargs: dict = None) -> plt.Axes:
        '''
        Plot the distribution of contour lengths for all subpaths for all paths in all particles. Uses Gaussian
        KDE to create a smooth distribution.

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
            inc_dist_kwargs = {'color': 'Blue', 'lw': 2, 'label': 'Included data'}
        if inc_fill_kwargs is None:
            inc_fill_kwargs = {'color': 'LightBlue', 'alpha': 0.5}
        if exc_dist_kwargs is None:
            exc_dist_kwargs = {'color': 'Gray', 'lw': 2, 'alpha': 0.5, 'label': 'Excluded data'}
        if exc_fill_kwargs is None:
            exc_fill_kwargs = {'color': 'LightGray', 'alpha': 0.5}
        if vline_kwargs is None:
            vline_kwargs = {'color': 'Blue', 'lw': 0.7, 'dashes': [8,3]}

        # Create a distribution of all the polymer branch lengths.
        xvals = np.linspace(0, self._contour_lengths.max(), n_points)
        kde = gaussian_kde(self._padded_contours[~np.isnan(self._padded_contours)])

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
    
    def plot_mean_squared_displacements(self,
                                        error_bars: bool = False,
                                        ax: plt.Axes = None,
                                        inc_kwargs: dict = None,
                                        exc_kwargs: dict = None,
                                        vline_kwargs: dict = None) -> plt.Axes:
        '''
        Plot the mean squared displacements for all particles.

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
            inc_kwargs = {'color': 'Blue', 'fmt': '.', 'ecolor': 'Blue', 'lw': 0.7, 'label': 'Fitted Data'}
        if exc_kwargs is None:
            exc_kwargs = {'color': 'Gray', 'fmt': '.', 'ecolor': 'Gray', 'lw': 0.7, 'label': 'Excluded Data'}
        if vline_kwargs is None:
            vline_kwargs = {'color': 'Blue', 'lw': 0.75, 'dashes': [8,3]}

        # If the error bars are set, the error is the standard error of the mean. Otherwise, the error is 0.
        if error_bars:
            # error = self._mean_squared_displacement_std
            ci = self._displacement_sq_confidence_intervals
            error = np.abs([self._mean_squared_displacements - ci[:,0], ci[:,1] - self._mean_squared_displacements])
        else:
            error = np.zeros((2, self._mean_squared_displacements.shape[0]))

        # If the minimum and maximum contour lengths are set, only color the region between the minimum and maximum, plot
        # the excluded regions in gray and plot vertical lines at the minimum and maximum contour lengths.
        if self._min_fitting_length != 0 or self._max_fitting_length != np.inf:
            
            # Get the mask for the xvalues in between the min and max contour lengths
            inbetween_mask = (self._contour_sampling >= self._min_fitting_length) * (self._contour_sampling <= self._max_fitting_length)

            # Plot the mean Tan-Tan correlation between the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[inbetween_mask],
                        self._mean_squared_displacements[inbetween_mask],
                        yerr = error[:,inbetween_mask],
                        **inc_kwargs)
            # Plot the mean Tan-Tan correlation outside the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[~inbetween_mask],
                        self._mean_squared_displacements[~inbetween_mask],
                        yerr = error[:,~inbetween_mask],
                        **exc_kwargs)
            
            # Draw the verical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_fitting_length, **vline_kwargs)
            ax.axvline(self._max_fitting_length, **vline_kwargs)

        else:
            # Plot the mean Tan-Tan correlation with error bars.
            ax.errorbar(self._contour_sampling,
                        self._mean_squared_displacements,
                        yerr = error,
                        **inc_kwargs)
        
        return ax
    
    def plot_mean_squared_displacements_fit(self,
                                            ax: plt.Axes = None,
                                            show_init: bool = False,
                                            fit_kwargs: dict = None,
                                            init_kwargs: dict = None,) -> plt.Axes:
        '''
        Plot the fitted R2 model of the mean squared displacements.

        Args:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object to plot the image on.
            show_init (bool):
                Whether or not to show the initial guess of the R2 model. Default is False.
            fit_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the fitted R2 model.
                Default is None.
            init_kwargs (dict):
                Keyword arguments to pass to matplotlib.pyplot.plot for the initial guess of the R2 model.
                Only used if show_init is True.
                Default is None.
        Returns:
            ax (matplotlib.axes.Axes):
                The matplotlib axis object.
        '''

        # Create the ax object if it is not set.
                # Create the ax object if it is not set.
        ax = ax or plt.gca()

        # Handle the default kwargs if they are none.
        if fit_kwargs is None:
            fit_kwargs = {'color': 'Red', 'lw': 1.5, 'label': 'Best Fit'}
        if init_kwargs is None:
            init_kwargs = {'color': 'Red', 'lw': 0.75, 'linestyle': '--', 'label': 'Initial Guess'}

        # Plot the fitted R2 model.
        x_ = self._contour_sampling
        y_ = self.__R2_model(self._contour_sampling, self._R2_fit_result.params['lp'].value)
        ax.plot(x_,y_,**fit_kwargs)

        # If show_init is true, also plot the initial guess.
        if show_init:
            ax.plot(self._contour_sampling,
                    self.__R2_model(self._contour_sampling, self._R2_fit_result.params['lp'].init_value),
                    **init_kwargs)

        return ax
    
    def plot_mean_tantan_correlations(self,
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
            inc_kwargs = {'color': 'Blue', 'fmt': '.', 'ecolor': 'Blue', 'lw': 0.7, 'label': 'Fitted Data'}
        if exc_kwargs is None:
            exc_kwargs = {'color': 'Gray', 'fmt': '.', 'ecolor': 'Gray', 'lw': 0.7, 'label': 'Excluded Data'}
        if vline_kwargs is None:
            vline_kwargs = {'color': 'Blue', 'lw': 0.75, 'dashes': [8,3]}

        # If the error bars are set, the error is the standard error of the mean. Otherwise, the error is 0.
        if error_bars:
            # error = self._mean_tantan_std
            ci = self._tantan_confidence_intervals
            error = np.abs([self._mean_tantan_correlations - ci[:,0], ci[:,1] - self._mean_tantan_correlations])
        else:
            error = np.zeros((2, self._mean_tantan_correlations.shape[0]))

        # If the minimum and maximum contour lengths are set, only color the region between the minimum and maximum, plot
        # the excluded regions in gray and plot vertical lines at the minimum and maximum contour lengths.
        if self._min_fitting_length != 0 or self._max_fitting_length != np.inf:
            
            # Get the mask for the xvalues in between the min and max contour lengths
            inbetween_mask = (self._contour_sampling >= self._min_fitting_length) * (self._contour_sampling <= self._max_fitting_length)

            
            # Plot the mean Tan-Tan correlation between the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[inbetween_mask],
                        self._mean_tantan_correlations[inbetween_mask],
                        yerr = error[:, inbetween_mask],
                        **inc_kwargs)
            # Plot the mean Tan-Tan correlation outside the minimum and maximum contour lengths with error bars.
            ax.errorbar(self._contour_sampling[~inbetween_mask],
                        self._mean_tantan_correlations[~inbetween_mask],
                        yerr = error[:, ~inbetween_mask],
                        **exc_kwargs)
            
            # Draw the verical lines for the minimum and maximum contour lengths.
            ax.axvline(self._min_fitting_length, **vline_kwargs)
            ax.axvline(self._max_fitting_length, **vline_kwargs)

        else:
            # Plot the mean Tan-Tan correlation with error bars.
            ax.errorbar(self._contour_sampling,
                        self._mean_tantan_correlations,
                        yerr = error,
                        **inc_kwargs)
            
        return ax

    def plot_mean_tantan_correlations_fit(self,
                                         ax: plt.Axes = None,
                                         show_init: bool = False,
                                         fit_kwargs: dict = None,
                                         init_kwargs: dict = None,) -> plt.Axes:
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
            fit_kwargs = {'color': 'Red', 'lw': 1.5, 'label': 'Best Fit'}
        if init_kwargs is None:
            init_kwargs = {'color': 'Red', 'lw': 0.75, 'linestyle': '--', 'label': 'Initial Guess'}

        # Plot the fitted exponential model.
        x_ = self._contour_sampling
        y_ = self.__exponential_model(self._contour_sampling, self._tantan_fit_result.params['lp'].value)
        ax.plot(x_,y_,**fit_kwargs)
        
        # If show_init is true, also plot the initial guess.
        if show_init:
            ax.plot(x_,
                    self.__exponential_model(self._contour_sampling, self._tantan_fit_result.params['lp'].init_value),
                    **init_kwargs)

        return ax
    
    def print_image_summary(self) -> None:
        '''
        Print a summary of the polymer image data.
        
        Args:
            None
        Returns:
            None
        '''
        print('Image Summary:')
        print(f'Number of Images:\t\t\t{len(self._images)}')
        print(f'Base Resolution:\t\t\t{self._metadata.get("base_resolution", 0):.1f} nm/pixel')
        if self._metadata.get('upscaled', False):
            print(f'Upscaled:\t\t\t\t{self._metadata.get("upscaled", False)}')
            print(f'Magnification Factor:\t\t\t{self._metadata.get("magnification", 0):.1f}')
            print(f'Interploation Order:\t\t\t{self._metadata.get('interpolation_order', 0)}')
            print(f'Upscaled Resolution:\t\t\t{self._resolution:.1f} nm/pixel')
    
    def print_classification_summary(self) -> None:
        '''
        Print a summary of the segmentation and classification for the polymer data.

        Args:
            None
        Returns:
            None
        '''
        print('Segmentation/Classification Summary:')
        print(f'All Particles:\t\t\t\t{self._num_particles.get("All", 0)}')
        print(f'Linear Particles:\t\t\t{self._num_particles.get("Linear", 0)}{'\t(Filtered Out)' if 'Linear' not in self._included_classifications else ''}')
        print(f'Branched Particles:\t\t\t{self._num_particles.get("Branched", 0)}{'\t(Filtered Out)' if 'Branched' not in self._included_classifications else ''}')
        print(f'Branched-Looped Particles:\t\t{self._num_particles.get("Branched-Looped", 0)}{'\t(Filtered Out)' if 'Branched-Looped' not in self._included_classifications else ''}')
        print(f'Looped Particles:\t\t\t{self._num_particles.get("Looped", 0)}{'\t(Filtered Out)' if 'Looped' not in self._included_classifications else ''}')
        print(f'Unknown Particles:\t\t\t{self._num_particles.get("Unknown", 0)}{'\t(Filtered Out)' if 'Unknown' not in self._included_classifications else ''}')

    def print_pl_summary(self) -> None:
        '''
        Print a summary of the persistence length calculations.

        Args:
            None
        Returns:
            None
        '''
        print('Persistence Length Summary:')
        if self._included_classifications == ['Linear', 'Branched', 'Looped', 'Branched-Looped', 'Unknown']:
            print('Included Classifications:\t\tAll')
        else:
            print(f'Included Classifications:\t\t{self._included_classifications}')
        print(f'Minimum Fitting Contour Length:\t\t{self._min_fitting_length:.1f} nm')
        print(f'Maximum Fitting Contour Length:\t\t{self._max_fitting_length:.1f} nm')
        try:
            print(f'R^2 lp:\t\t\t\t\t{self._R2_fit_result.params["lp"].value * self._resolution:.1f} +/- {self._R2_fit_result.params["lp"].stderr * self._resolution:.1f} nm')
            print(f'R^2 Reduced Chi^2:\t\t\t{self._R2_fit_result.redchi:.2f}')
        except AttributeError:
            pass
        try:
            print(f'Tan-Tan Correlation lp:\t\t\t{self._tantan_fit_result.params["lp"].value * self._resolution:.1f} +/- {self._tantan_fit_result.params["lp"].stderr * self._resolution:.1f} nm')
            print(f'Tan-Tan Correlation Reduced Chi^2:\t{self._tantan_fit_result.redchi:.2f}')
        except AttributeError:
            pass

    def print_summary(self) -> None:
        '''
        Print a summary of the polymer data.

        Args:
            None
        Returns:
            None
        '''
        print('Polymer Data Summary')
        print('-'*64)
        self.print_image_summary()
        print('-'*64)
        self.print_classification_summary()
        print('-'*64)
        self.print_pl_summary()
        
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
    def num_particles(self) -> Dict[str, int]:
        '''
        Dictionary of the number of particles for each classification.
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
    def included_classifications(self) -> list[str]:
        """
        list[str]: The current list of particle classifications that are included in
        the analysis (i.e., after filtering).
        """
        return self._included_classifications

    @property
    def mean_squared_displacements(self) -> np.ndarray:
        '''
        The mean squared displacements of all particles. Calculated by taking the mean of the squared displacements for each
        path in each particle. 
        '''
        return self._mean_squared_displacements

    @property
    def mean_tantan_correlation(self) -> np.ndarray:
        '''
        The mean Tan-Tan correlation of all particles. Calculated by taking the mean of the Tan-Tan correlation for each
        path in each particle. 
        '''
        return self._mean_tantan_correlations
    
    @property
    def min_fitting_length(self) -> float:
        """
        float: The minimum contour length (in nm) used when fitting the
        persistence-length models.
        """
        return self._min_fitting_length

    @property
    def max_fitting_length(self) -> float:
        """
        float: The maximum contour length (in nm) used when fitting the
        persistence-length models.
        """
        return self._max_fitting_length
    
    @property
    def R2_fit_result(self) -> lmfit.model.ModelResult:
        '''
        The lmfit model result of the end to end distance squared fit.
        '''
        return self._R2_fit_result
    
    @property
    def tantan_fit_result(self) -> lmfit.model.ModelResult:
        '''
        The lmfit model result of the Tan-Tan correlation fit.
        '''
        return self._tantan_fit_result