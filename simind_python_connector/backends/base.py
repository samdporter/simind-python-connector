"""
Base interfaces for SIRF and STIR Python backends.

This module defines abstract base classes that provide a unified interface
for working with both SIRF and STIR Python libraries.
"""

from abc import ABC, abstractmethod

import numpy as np


class ImageDataInterface(ABC):
    """Abstract interface for image data objects.

    Includes SIRF ``ImageData`` and STIR ``FloatVoxelsOnCartesianGrid`` types.
    """

    @classmethod
    @abstractmethod
    def read_from_file(cls, filepath: str) -> "ImageDataInterface":
        """Read image data from file.

        Args:
            filepath: Path to image file (.hv)

        Returns:
            ImageDataInterface: Wrapped image object
        """

    @abstractmethod
    def write(self, filepath: str) -> None:
        """Write image data to file.

        Args:
            filepath: Output file path
        """

    @abstractmethod
    def fill(self, data: np.ndarray | float) -> None:
        """Fill image with data.

        Args:
            data: Numpy array or scalar value
        """

    @abstractmethod
    def clone(self) -> "ImageDataInterface":
        """Create a deep copy of the image.

        Returns:
            ImageDataInterface: Cloned image object
        """

    @abstractmethod
    def get_uniform_copy(self, value: float) -> "ImageDataInterface":
        """Create a uniform copy with specified value.

        Args:
            value: Value to fill the image with

        Returns:
            ImageDataInterface: New uniform image
        """

    @abstractmethod
    def dimensions(self) -> tuple[int, int, int]:
        """Get image dimensions.

        Returns:
            Tuple[int, int, int]: (z, y, x) dimensions
        """

    @abstractmethod
    def voxel_sizes(self) -> tuple[float, float, float]:
        """Get voxel sizes in mm.

        Returns:
            Tuple[float, float, float]: (z, y, x) voxel sizes
        """

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Image data as numpy array
        """

    @abstractmethod
    def sum(self) -> float:
        """Sum all values in the image.

        Returns:
            float: Sum of all voxel values
        """

    @abstractmethod
    def max(self) -> float:
        """Return the maximum voxel value.

        Returns:
            float: Maximum voxel value
        """

    @abstractmethod
    def maximum(self, value: float) -> None:
        """Apply element-wise maximum (clipping).

        Args:
            value: Minimum value to clip to
        """

    @property
    @abstractmethod
    def native_object(self):
        """Get the underlying native object.

        Returns either SIRF ``ImageData`` or STIR
        ``FloatVoxelsOnCartesianGrid``.
        """

    # Arithmetic operations
    @abstractmethod
    def __add__(self, other):
        """Add two images or image and scalar."""

    @abstractmethod
    def __sub__(self, other):
        """Subtract two images or image and scalar."""

    @abstractmethod
    def __mul__(self, other):
        """Multiply image by scalar or element-wise."""

    @abstractmethod
    def __truediv__(self, other):
        """Divide image by scalar or element-wise."""

    @abstractmethod
    def __radd__(self, other):
        """Right-hand addition."""

    @abstractmethod
    def __rsub__(self, other):
        """Right-hand subtraction."""

    @abstractmethod
    def __rmul__(self, other):
        """Right-hand multiplication."""

    @abstractmethod
    def __rtruediv__(self, other):
        """Right-hand division."""

    @abstractmethod
    def __neg__(self):
        """Negate image."""

    @abstractmethod
    def sapyb(
        self, a: float, y: "ImageDataInterface", b: float
    ) -> "ImageDataInterface":
        """Compute self * a + y * b (element-wise), modifying self in place.

        This method modifies self in place: self = self * a + y * b

        Args:
            a: Scalar multiplier for self
            y: Other image
            b: Scalar multiplier for y

        Returns:
            ImageDataInterface: self after modification (for chaining)
        """


class AcquisitionDataInterface(ABC):
    """Abstract interface for acquisition data (AcquisitionData or ProjData)."""

    @classmethod
    @abstractmethod
    def read_from_file(cls, filepath: str) -> "AcquisitionDataInterface":
        """Read acquisition data from file.

        Args:
            filepath: Path to header file (.hs)

        Returns:
            AcquisitionDataInterface: Wrapped acquisition object
        """

    @abstractmethod
    def write(self, filepath: str) -> None:
        """Write acquisition data to file.

        Args:
            filepath: Output file path
        """

    @abstractmethod
    def fill(self, data: np.ndarray | float) -> None:
        """Fill acquisition data.

        Args:
            data: Numpy array or scalar value
        """

    @abstractmethod
    def clone(self) -> "AcquisitionDataInterface":
        """Create a deep copy.

        Returns:
            AcquisitionDataInterface: Cloned object
        """

    @abstractmethod
    def get_uniform_copy(self, value: float) -> "AcquisitionDataInterface":
        """Create a uniform copy with specified value.

        Args:
            value: Value to fill the acquisition data with

        Returns:
            AcquisitionDataInterface: New uniform acquisition data
        """

    @abstractmethod
    def dimensions(self) -> tuple:
        """Get acquisition data dimensions.

        Returns:
            Tuple: Dimensions (varies by format)
        """

    @property
    def shape(self) -> tuple:
        """Get acquisition data shape (alias for dimensions()).

        Returns:
            Tuple: Dimensions (varies by format)
        """
        return self.dimensions()

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """Convert to numpy array.

        Returns:
            np.ndarray: Acquisition data as numpy array
        """

    @abstractmethod
    def sum(self) -> float:
        """Sum all values.

        Returns:
            float: Sum of all counts
        """

    @abstractmethod
    def max(self) -> float:
        """Return the maximum bin value.

        Returns:
            float: Maximum bin value
        """

    @abstractmethod
    def get_info(self) -> str:
        """Get metadata information.

        Returns:
            str: Metadata string
        """

    @abstractmethod
    def get_energy_window_bounds(self) -> tuple[float, float]:
        """Get the energy window bounds for the acquisition.

        Returns:
            Tuple[float, float]: (lower_threshold, upper_threshold) in keV
        """

    @property
    @abstractmethod
    def native_object(self):
        """Get the underlying native object (AcquisitionData or ProjData)."""

    # Arithmetic operations
    @abstractmethod
    def __add__(self, other):
        """Add two acquisition data or acquisition data and scalar."""

    @abstractmethod
    def __sub__(self, other):
        """Subtract two acquisition data or acquisition data and scalar."""

    @abstractmethod
    def __mul__(self, other):
        """Multiply acquisition data by scalar or element-wise."""

    @abstractmethod
    def __truediv__(self, other):
        """Divide acquisition data by scalar or element-wise."""

    @abstractmethod
    def __radd__(self, other):
        """Right-hand addition."""

    @abstractmethod
    def __rsub__(self, other):
        """Right-hand subtraction."""

    @abstractmethod
    def __rmul__(self, other):
        """Right-hand multiplication."""

    @abstractmethod
    def __rtruediv__(self, other):
        """Right-hand division."""

    @abstractmethod
    def __neg__(self):
        """Negate acquisition data."""

    @abstractmethod
    def sapyb(
        self, a: float, y: "AcquisitionDataInterface", b: float
    ) -> "AcquisitionDataInterface":
        """Compute self * a + y * b (element-wise), modifying self in place.

        This method modifies self in place: self = self * a + y * b

        Args:
            a: Scalar multiplier for self
            y: Other acquisition data
            b: Scalar multiplier for y

        Returns:
            AcquisitionDataInterface: self after modification (for chaining)
        """
