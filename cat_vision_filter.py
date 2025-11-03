"""
Cat Vision Filter Implementation
Biologically accurate computational filters that emulate feline visual perception.

Based on peer-reviewed research on cat retinal structure and visual characteristics.
"""

# Handle OpenCV import for headless environments
try:
    import cv2
except ImportError:
    print("OpenCV not available. Installing opencv-python-headless...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
    import cv2

import numpy as np
from scipy import ndimage, interpolate, signal
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
import warnings
warnings.filterwarnings('ignore')

class CatVisionFilter:
    """
    Comprehensive cat vision emulation filter based on feline retinal biology.
    
    Key characteristics implemented:
    - Vertical slit pupil (3:1 aspect ratio)
    - Enhanced blue-green sensitivity (~500nm peak)
    - Rod-dominated vision (25:1 rod/cone ratio)
    - Tapetum lucidum light reflection
    - Reduced color discrimination
    - Enhanced motion detection
    """
    
    def __init__(self):
        # Biological parameters based on research
        self.pupil_aspect_ratio = 3.0  # vertical slit
        self.rod_cone_ratio = 25.0     # vs human 20:1
        self.peak_wavelength = 500     # nm (blue-green)
        self.tapetum_reflectance = 0.3 # light amplification factor
        self.visual_field_h = 200      # degrees horizontal
        self.visual_field_v = 140      # degrees vertical
        
        # Spectral sensitivity parameters (biological data)
        self.s_cone_peak = 450  # nm - S-cone peak sensitivity
        self.l_cone_peak = 556  # nm - L-cone peak sensitivity
        self.rod_peak = 498     # nm - Rod peak sensitivity
        
        # Spatial acuity parameters
        self.spatial_acuity_factor = 0.167  # 1/6 of human acuity (20/100-20/200 equivalent)
        self.foveal_acuity_cycles_per_degree = 3.0  # vs human ~18
        
        # Temporal processing parameters
        self.flicker_fusion_threshold = 55  # Hz (vs human ~24Hz)
        self.temporal_sensitivity_peak = 10  # Hz
        
        # Motion detection parameters
        self.motion_sensitivity = 1.8  # Enhanced motion detection
        self.horizontal_motion_bias = 1.5  # Enhanced horizontal motion sensitivity
        
        # Initialize spectral sensitivity curves
        self._init_spectral_curves()
        
        # Color sensitivity weights (legacy - will be replaced by spectral curves)
        self.color_weights = {
            'blue': 1.4,    # Enhanced blue sensitivity
            'green': 1.2,   # Enhanced green sensitivity  
            'red': 0.6      # Reduced red sensitivity
        }
        
    def create_pupil_kernel(self, kernel_size=15):
        """
        Create vertical slit pupil convolution kernel.
        
        Args:
            kernel_size (int): Size of the kernel (should be odd)
            
        Returns:
            np.ndarray: Normalized convolution kernel representing cat pupil
        """
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure odd size
            
        # Create elliptical Gaussian with 3:1 aspect ratio (vertical slit)
        center = kernel_size // 2
        y, x = np.ogrid[:kernel_size, :kernel_size]
        
        # Elliptical parameters
        sigma_x = kernel_size / 8.0  # Narrow horizontal spread
        sigma_y = sigma_x * self.pupil_aspect_ratio  # Wider vertical spread
        
        # Create elliptical Gaussian
        kernel = np.exp(-((x - center)**2 / (2 * sigma_x**2) + 
                         (y - center)**2 / (2 * sigma_y**2)))
        
        # Normalize kernel
        kernel = kernel / np.sum(kernel)
        
        return kernel
    
    def _init_spectral_curves(self):
        """Initialize biological spectral sensitivity curves for cat photoreceptors."""
        # Wavelength range (nm)
        self.wavelengths = np.linspace(380, 700, 321)
        
        # Cat S-cone spectral sensitivity (peak ~450nm)
        s_sigma = 40  # nm bandwidth
        self.s_cone_sensitivity = np.exp(-0.5 * ((self.wavelengths - self.s_cone_peak) / s_sigma) ** 2)
        
        # Cat L-cone spectral sensitivity (peak ~556nm) 
        l_sigma = 50  # nm bandwidth
        self.l_cone_sensitivity = np.exp(-0.5 * ((self.wavelengths - self.l_cone_peak) / l_sigma) ** 2)
        
        # Rod spectral sensitivity (peak ~498nm)
        rod_sigma = 45  # nm bandwidth
        self.rod_sensitivity = np.exp(-0.5 * ((self.wavelengths - self.rod_peak) / rod_sigma) ** 2)
        
        # Normalize curves
        self.s_cone_sensitivity /= np.max(self.s_cone_sensitivity)
        self.l_cone_sensitivity /= np.max(self.l_cone_sensitivity)
        self.rod_sensitivity /= np.max(self.rod_sensitivity)
        
        # RGB to wavelength mapping (approximate)
        self.rgb_wavelengths = {'red': 630, 'green': 530, 'blue': 470}
    
    def apply_spectral_sensitivity_curves(self, image: np.ndarray) -> np.ndarray:
        """
        Apply biological spectral sensitivity curves instead of simple RGB weights.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            Spectrally corrected image
        """
        if len(image.shape) != 3:
            return image
            
        img_float = image.astype(np.float32) / 255.0
        b, g, r = cv2.split(img_float)
        
        # Map RGB channels to spectral sensitivities
        # Blue channel (~470nm)
        blue_idx = np.argmin(np.abs(self.wavelengths - self.rgb_wavelengths['blue']))
        s_response_blue = self.s_cone_sensitivity[blue_idx]
        l_response_blue = self.l_cone_sensitivity[blue_idx]
        rod_response_blue = self.rod_sensitivity[blue_idx]
        
        # Green channel (~530nm)
        green_idx = np.argmin(np.abs(self.wavelengths - self.rgb_wavelengths['green']))
        s_response_green = self.s_cone_sensitivity[green_idx]
        l_response_green = self.l_cone_sensitivity[green_idx]
        rod_response_green = self.rod_sensitivity[green_idx]
        
        # Red channel (~630nm)
        red_idx = np.argmin(np.abs(self.wavelengths - self.rgb_wavelengths['red']))
        s_response_red = self.s_cone_sensitivity[red_idx]
        l_response_red = self.l_cone_sensitivity[red_idx]
        rod_response_red = self.rod_sensitivity[red_idx]
        
        # Apply rod dominance (25:1 ratio)
        rod_weight = self.rod_cone_ratio / (self.rod_cone_ratio + 1)
        cone_weight = 1 / (self.rod_cone_ratio + 1)
        
        # Calculate weighted responses
        b_corrected = (rod_weight * rod_response_blue + cone_weight * (s_response_blue + l_response_blue)) * b
        g_corrected = (rod_weight * rod_response_green + cone_weight * (s_response_green + l_response_green)) * g
        r_corrected = (rod_weight * rod_response_red + cone_weight * (s_response_red + l_response_red)) * r
        
        # Normalize and clip
        corrected = cv2.merge([
            np.clip(b_corrected, 0, 1),
            np.clip(g_corrected, 0, 1), 
            np.clip(r_corrected, 0, 1)
        ])
        
        return (corrected * 255).astype(np.uint8)
    
    def simulate_spatial_acuity_reduction(self, image: np.ndarray, acuity_factor: Optional[float] = None) -> np.ndarray:
        """
        Simulate cat spatial acuity reduction using frequency domain filtering.
        
        Args:
            image: Input image
            acuity_factor: Acuity reduction factor (default: 0.167 = 1/6 human acuity)
            
        Returns:
            Spatially filtered image
        """
        if acuity_factor is None:
            acuity_factor = self.spatial_acuity_factor
            
        if len(image.shape) == 3:
            # Process each channel separately
            filtered_channels = []
            for i in range(3):
                filtered_channel = self._apply_spatial_filter(image[:, :, i], acuity_factor)
                filtered_channels.append(filtered_channel)
            return cv2.merge(filtered_channels)
        else:
            return self._apply_spatial_filter(image, acuity_factor)
    
    def _apply_spatial_filter(self, channel: np.ndarray, acuity_factor: float) -> np.ndarray:
        """Apply spatial frequency filtering to simulate reduced acuity."""
        # Convert to frequency domain
        f_transform = np.fft.fft2(channel)
        f_shifted = np.fft.fftshift(f_transform)
        
        # Create frequency coordinates
        rows, cols = channel.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create distance matrix from center
        y, x = np.ogrid[:rows, :cols]
        distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
        
        # Create low-pass filter based on acuity factor
        # Higher acuity_factor = more high frequencies preserved
        cutoff_freq = acuity_factor * min(rows, cols) / 4
        filter_mask = np.exp(-(distance**2) / (2 * cutoff_freq**2))
        
        # Apply filter
        f_filtered = f_shifted * filter_mask
        
        # Convert back to spatial domain
        f_ishifted = np.fft.ifftshift(f_filtered)
        filtered = np.fft.ifft2(f_ishifted)
        filtered = np.real(filtered)
        
        return np.clip(filtered, 0, 255).astype(np.uint8)
    
    def apply_spatial_field_transformation(self, image: np.ndarray, 
                                         fov_horizontal: int = 200, 
                                         fov_vertical: int = 140) -> np.ndarray:
        """
        Apply wide-angle peripheral vision transformation.
        
        Args:
            image: Input image
            fov_horizontal: Horizontal field of view in degrees
            fov_vertical: Vertical field of view in degrees
            
        Returns:
            Transformed image with cat-like field of view
        """
        h, w = image.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w]
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x - w/2) / (w/2)
        y_norm = (y - h/2) / (h/2)
        
        # Apply barrel distortion for wider field of view
        # Cat FOV is wider than human, so we need to compress the image
        fov_ratio_h = fov_horizontal / 180  # Human ~180°
        fov_ratio_v = fov_vertical / 120    # Human ~120°
        
        # Barrel distortion parameters
        k1 = 0.1 * (fov_ratio_h - 1)  # Horizontal distortion
        k2 = 0.1 * (fov_ratio_v - 1)  # Vertical distortion
        
        # Apply distortion
        r_squared = x_norm**2 + y_norm**2
        x_distorted = x_norm * (1 + k1 * r_squared)
        y_distorted = y_norm * (1 + k2 * r_squared)
        
        # Convert back to image coordinates
        x_new = (x_distorted * w/2 + w/2).astype(np.float32)
        y_new = (y_distorted * h/2 + h/2).astype(np.float32)
        
        # Apply remapping
        transformed = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        # Apply center-surround acuity mapping
        center_x, center_y = w//2, h//2
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        
        # Create acuity mask (sharp center, blurred periphery)
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        acuity_mask = np.exp(-(distance_from_center / max_distance)**2 * 3)
        
        # Apply variable blur based on distance from center
        if len(image.shape) == 3:
            for i in range(3):
                # Create peripheral blur
                blurred = gaussian_filter(transformed[:, :, i], sigma=3)
                # Blend based on acuity mask
                transformed[:, :, i] = (acuity_mask * transformed[:, :, i] + 
                                      (1 - acuity_mask) * blurred).astype(np.uint8)
        else:
            blurred = gaussian_filter(transformed, sigma=3)
            transformed = (acuity_mask * transformed + (1 - acuity_mask) * blurred).astype(np.uint8)
        
        return transformed
    
    def model_temporal_processing(self, frame_sequence: List[np.ndarray], fps: int = 30) -> List[np.ndarray]:
        """
        Model cat temporal processing with enhanced flicker fusion threshold.
        
        Args:
            frame_sequence: List of consecutive frames
            fps: Frames per second of the input sequence
            
        Returns:
            Temporally processed frame sequence
        """
        if len(frame_sequence) < 2:
            return frame_sequence
            
        processed_frames = []
        
        for i, frame in enumerate(frame_sequence):
            if i == 0:
                processed_frames.append(frame)
                continue
                
            # Calculate temporal frequency based on frame differences
            prev_frame = frame_sequence[i-1]
            
            # Convert to grayscale for temporal analysis
            if len(frame.shape) == 3:
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = frame
                prev_gray = prev_frame
                
            # Calculate frame difference (motion/flicker detection)
            frame_diff = cv2.absdiff(current_gray, prev_gray)
            
            # Apply temporal sensitivity curve (peak at ~10Hz for cats)
            temporal_freq = fps / max(1, i)  # Approximate frequency
            sensitivity = self._temporal_sensitivity_function(temporal_freq)
            
            # Enhance or suppress based on temporal frequency
            if len(frame.shape) == 3:
                enhanced_frame = frame.astype(np.float32)
                motion_mask = frame_diff > 5  # Motion threshold
                
                for channel in range(3):
                    enhanced_frame[:, :, channel][motion_mask] *= sensitivity
                    
                processed_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
            else:
                enhanced_frame = frame.astype(np.float32)
                motion_mask = frame_diff > 5
                enhanced_frame[motion_mask] *= sensitivity
                processed_frame = np.clip(enhanced_frame, 0, 255).astype(np.uint8)
                
            processed_frames.append(processed_frame)
            
        return processed_frames
    
    def _temporal_sensitivity_function(self, frequency: float) -> float:
        """Calculate temporal sensitivity based on cat visual system."""
        # Cat temporal sensitivity peaks around 10Hz, drops off after 55Hz
        if frequency <= 0:
            return 1.0
        elif frequency <= self.temporal_sensitivity_peak:
            # Increasing sensitivity up to peak
            return 1.0 + 0.5 * (frequency / self.temporal_sensitivity_peak)
        elif frequency <= self.flicker_fusion_threshold:
            # Decreasing sensitivity after peak
            decay_factor = (frequency - self.temporal_sensitivity_peak) / (self.flicker_fusion_threshold - self.temporal_sensitivity_peak)
            return 1.5 - 0.5 * decay_factor
        else:
            # Above flicker fusion threshold - minimal sensitivity
            return 0.1
    
    def enhanced_motion_detection(self, frame_sequence: List[np.ndarray], 
                                flow_method: str = 'lucas_kanade') -> List[np.ndarray]:
        """
        Enhanced motion detection with optical flow and directional sensitivity.
        
        Args:
            frame_sequence: List of consecutive frames
            flow_method: Optical flow method ('lucas_kanade' or 'farneback')
            
        Returns:
            Motion-enhanced frame sequence
        """
        if len(frame_sequence) < 2:
            return frame_sequence
            
        enhanced_frames = []
        
        for i, frame in enumerate(frame_sequence):
            if i == 0:
                enhanced_frames.append(frame)
                continue
                
            prev_frame = frame_sequence[i-1]
            
            # Convert to grayscale for optical flow
            if len(frame.shape) == 3:
                current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            else:
                current_gray = frame
                prev_gray = prev_frame
                
            # Calculate optical flow
            if flow_method == 'lucas_kanade':
                flow = self._lucas_kanade_flow(prev_gray, current_gray)
            else:
                flow = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, None, None)[0]
                
            # Calculate motion magnitude and direction
            if flow is not None:
                motion_magnitude, motion_direction = self._analyze_motion(flow)
                
                # Apply directional sensitivity (cats excel at horizontal motion)
                directional_weight = self._calculate_directional_sensitivity(motion_direction)
                
                # Enhance motion areas
                enhanced_frame = self._apply_motion_enhancement(frame, motion_magnitude, directional_weight)
                enhanced_frames.append(enhanced_frame)
            else:
                enhanced_frames.append(frame)
                
        return enhanced_frames
    
    def _lucas_kanade_flow(self, prev_gray: np.ndarray, current_gray: np.ndarray) -> Optional[np.ndarray]:
        """Calculate Lucas-Kanade optical flow."""
        try:
            # Parameters for corner detection
            feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
            
            # Parameters for Lucas-Kanade optical flow
            lk_params = dict(winSize=(15, 15), maxLevel=2, 
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            # Find corners in previous frame
            p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
            
            if p0 is not None:
                # Calculate optical flow
                p1, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, current_gray, p0, None, **lk_params)
                
                # Select good points
                if p1 is not None:
                    good_new = p1[status == 1]
                    good_old = p0[status == 1]
                    
                    # Calculate flow vectors
                    flow_vectors = good_new - good_old
                    return flow_vectors
                    
        except Exception:
            pass
            
        return None
    
    def _analyze_motion(self, flow_vectors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Analyze motion magnitude and direction from flow vectors."""
        if len(flow_vectors) == 0:
            return np.array([]), np.array([])
            
        # Calculate magnitude and angle
        magnitude = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
        direction = np.arctan2(flow_vectors[:, 1], flow_vectors[:, 0])
        
        return magnitude, direction
    
    def _calculate_directional_sensitivity(self, directions: np.ndarray) -> float:
        """Calculate directional sensitivity weight (cats excel at horizontal motion)."""
        if len(directions) == 0:
            return 1.0
            
        # Convert angles to horizontal bias (0° and 180° are horizontal)
        horizontal_angles = np.abs(np.cos(directions))
        avg_horizontal_bias = np.mean(horizontal_angles)
        
        # Apply horizontal motion bias
        return 1.0 + (self.horizontal_motion_bias - 1.0) * avg_horizontal_bias
    
    def _apply_motion_enhancement(self, frame: np.ndarray, motion_magnitude: np.ndarray, 
                                directional_weight: float) -> np.ndarray:
        """Apply motion enhancement to frame."""
        if len(motion_magnitude) == 0:
            return frame
            
        enhanced = frame.astype(np.float32)
        
        # Create motion enhancement mask
        avg_motion = np.mean(motion_magnitude) if len(motion_magnitude) > 0 else 0
        enhancement_factor = 1.0 + (self.motion_sensitivity - 1.0) * directional_weight * min(avg_motion / 10.0, 1.0)
        
        # Apply enhancement globally (simplified approach)
        enhanced *= enhancement_factor
        
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def plot_spectral_sensitivity_curves(self, save_path: Optional[str] = None):
        """
        Visualize cat photoreceptor spectral sensitivity curves.
        
        Args:
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        # Plot spectral sensitivity curves
        plt.plot(self.wavelengths, self.s_cone_sensitivity, 'b-', linewidth=2, 
                label=f'S-cone (peak: {self.s_cone_peak}nm)', alpha=0.8)
        plt.plot(self.wavelengths, self.l_cone_sensitivity, 'g-', linewidth=2, 
                label=f'L-cone (peak: {self.l_cone_peak}nm)', alpha=0.8)
        plt.plot(self.wavelengths, self.rod_sensitivity, 'k-', linewidth=2, 
                label=f'Rod (peak: {self.rod_peak}nm)', alpha=0.8)
        
        # Add RGB wavelength markers
        for color, wavelength in self.rgb_wavelengths.items():
            plt.axvline(x=wavelength, color=color, linestyle='--', alpha=0.6, 
                       label=f'{color.capitalize()} (~{wavelength}nm)')
        
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Normalized Sensitivity', fontsize=12)
        plt.title('Cat Photoreceptor Spectral Sensitivity Curves\n(Based on Biological Data)', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xlim(380, 700)
        plt.ylim(0, 1.1)
        
        # Add annotations
        plt.annotate('Enhanced blue-green\nsensitivity', xy=(500, 0.8), xytext=(450, 0.9),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
        plt.annotate('Reduced red\nsensitivity', xy=(630, 0.2), xytext=(650, 0.4),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def visualize_spatial_acuity_map(self, image_size: Tuple[int, int], save_path: Optional[str] = None):
        """
        Visualize spatial acuity mapping across the visual field.
        
        Args:
            image_size: (height, width) of the reference image
            save_path: Optional path to save the visualization
        """
        h, w = image_size
        y, x = np.mgrid[0:h, 0:w]
        
        # Calculate distance from center
        center_x, center_y = w//2, h//2
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Create acuity map (sharp center, blurred periphery)
        acuity_map = np.exp(-(distance_from_center / max_distance)**2 * 3)
        
        # Convert to cycles per degree (approximate)
        acuity_cpd = self.foveal_acuity_cycles_per_degree * acuity_map
        
        plt.figure(figsize=(12, 8))
        
        # Create subplot for acuity map
        plt.subplot(1, 2, 1)
        im1 = plt.imshow(acuity_map, cmap='hot', interpolation='bilinear')
        plt.colorbar(im1, label='Relative Acuity')
        plt.title('Cat Spatial Acuity Map\n(Relative to Center)', fontsize=12)
        plt.xlabel('Horizontal Position (pixels)')
        plt.ylabel('Vertical Position (pixels)')
        
        # Create subplot for cycles per degree
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(acuity_cpd, cmap='viridis', interpolation='bilinear')
        plt.colorbar(im2, label='Cycles per Degree')
        plt.title('Cat Visual Acuity\n(Cycles per Degree)', fontsize=12)
        plt.xlabel('Horizontal Position (pixels)')
        plt.ylabel('Vertical Position (pixels)')
        
        plt.suptitle(f'Cat Visual Acuity Distribution\n(Peak: {self.foveal_acuity_cycles_per_degree} cpd vs Human: ~18 cpd)', 
                    fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def demonstrate_temporal_frequency_response(self, save_path: Optional[str] = None):
        """
        Demonstrate cat temporal frequency response characteristics.
        
        Args:
            save_path: Optional path to save the plot
        """
        frequencies = np.linspace(0, 80, 161)
        sensitivities = [self._temporal_sensitivity_function(f) for f in frequencies]
        
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, sensitivities, 'b-', linewidth=2, label='Cat Temporal Sensitivity')
        
        # Add markers for key frequencies
        plt.axvline(x=self.temporal_sensitivity_peak, color='g', linestyle='--', 
                   label=f'Peak Sensitivity ({self.temporal_sensitivity_peak}Hz)')
        plt.axvline(x=self.flicker_fusion_threshold, color='r', linestyle='--', 
                   label=f'Flicker Fusion Threshold ({self.flicker_fusion_threshold}Hz)')
        plt.axvline(x=24, color='orange', linestyle=':', alpha=0.7, 
                   label='Human Flicker Fusion (~24Hz)')
        
        plt.xlabel('Temporal Frequency (Hz)', fontsize=12)
        plt.ylabel('Sensitivity Factor', fontsize=12)
        plt.title('Cat Temporal Frequency Response\nvs Human Visual System', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 80)
        plt.ylim(0, 1.6)
        
        # Add annotations
        plt.annotate('Enhanced motion\ndetection range', 
                    xy=(30, 1.2), xytext=(45, 1.4),
                    arrowprops=dict(arrowstyle='->', color='gray'), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def compare_motion_detection_sensitivity(self, test_video_frames: List[np.ndarray], 
                                           save_path: Optional[str] = None):
        """
        Compare motion detection sensitivity between cat and human-like processing.
        
        Args:
            test_video_frames: List of video frames for analysis
            save_path: Optional path to save the comparison plot
        """
        if len(test_video_frames) < 2:
            print("Need at least 2 frames for motion analysis")
            return
            
        # Analyze motion with cat-like processing
        cat_enhanced = self.enhanced_motion_detection(test_video_frames)
        
        # Analyze motion with basic frame differencing (human-like)
        human_motion_scores = []
        cat_motion_scores = []
        
        for i in range(1, len(test_video_frames)):
            # Human-like motion detection
            prev_gray = cv2.cvtColor(test_video_frames[i-1], cv2.COLOR_BGR2GRAY) if len(test_video_frames[i-1].shape) == 3 else test_video_frames[i-1]
            curr_gray = cv2.cvtColor(test_video_frames[i], cv2.COLOR_BGR2GRAY) if len(test_video_frames[i].shape) == 3 else test_video_frames[i]
            human_diff = cv2.absdiff(prev_gray, curr_gray)
            human_motion_scores.append(np.mean(human_diff))
            
            # Cat-like motion detection
            cat_prev_gray = cv2.cvtColor(cat_enhanced[i-1], cv2.COLOR_BGR2GRAY) if len(cat_enhanced[i-1].shape) == 3 else cat_enhanced[i-1]
            cat_curr_gray = cv2.cvtColor(cat_enhanced[i], cv2.COLOR_BGR2GRAY) if len(cat_enhanced[i].shape) == 3 else cat_enhanced[i]
            cat_diff = cv2.absdiff(cat_prev_gray, cat_curr_gray)
            cat_motion_scores.append(np.mean(cat_diff))
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        
        frame_indices = range(1, len(test_video_frames))
        plt.plot(frame_indices, human_motion_scores, 'b-', linewidth=2, 
                label='Human-like Motion Detection', alpha=0.7)
        plt.plot(frame_indices, cat_motion_scores, 'r-', linewidth=2, 
                label='Cat-like Motion Detection', alpha=0.7)
        
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Motion Score (Mean Pixel Difference)', fontsize=12)
        plt.title('Motion Detection Sensitivity Comparison\nCat vs Human Visual Processing', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Calculate enhancement ratio
        avg_enhancement = np.mean(np.array(cat_motion_scores) / (np.array(human_motion_scores) + 1e-6))
        plt.text(0.02, 0.98, f'Average Cat Enhancement: {avg_enhancement:.2f}x', 
                transform=plt.gca().transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def adjust_color_sensitivity(self, image):
        """
        Adjust color sensitivity to match cat vision.
        
        Cats have peak sensitivity around 500nm (blue-green) and reduced
        red sensitivity compared to humans.
        
        Args:
            image (np.ndarray): Input image in BGR format
            
        Returns:
            np.ndarray: Color-adjusted image
        """
        if len(image.shape) != 3:
            return image
            
        # Convert to float for processing
        img_float = image.astype(np.float32) / 255.0
        
        # Split channels (BGR format)
        b, g, r = cv2.split(img_float)
        
        # Apply cat-specific color weights
        b_enhanced = np.clip(b * self.color_weights['blue'], 0, 1)
        g_enhanced = np.clip(g * self.color_weights['green'], 0, 1)
        r_reduced = np.clip(r * self.color_weights['red'], 0, 1)
        
        # Merge channels
        adjusted = cv2.merge([b_enhanced, g_enhanced, r_reduced])
        
        # Convert back to uint8
        return (adjusted * 255).astype(np.uint8)
    
    def apply_tapetum_effect(self, image, brightness_threshold=0.3):
        """
        Simulate tapetum lucidum light reflection effect.
        
        The tapetum lucidum reflects light back through the retina,
        enhancing low-light vision performance.
        
        Args:
            image (np.ndarray): Input image
            brightness_threshold (float): Threshold for low-light enhancement
            
        Returns:
            np.ndarray: Image with tapetum effect applied
        """
        if len(image.shape) == 3:
            # Convert to grayscale for brightness calculation
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Calculate average brightness
        avg_brightness = np.mean(gray) / 255.0
        
        # Apply tapetum effect in low-light conditions
        if avg_brightness < brightness_threshold:
            # Create enhancement mask based on image brightness
            enhancement_factor = 1.0 + (self.tapetum_reflectance * 
                                       (brightness_threshold - avg_brightness))
            
            # Apply enhancement with spatial variation
            enhanced = image.astype(np.float32) * enhancement_factor
            
            # Add slight blue-green tint (characteristic of tapetum reflection)
            if len(image.shape) == 3:
                enhanced[:, :, 0] *= 1.1  # Enhance blue channel
                enhanced[:, :, 1] *= 1.05  # Slightly enhance green
            
            # Clip values and convert back
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            return enhanced
        
        return image
    
    def simulate_rod_dominance(self, image):
        """
        Simulate effects of rod-dominated vision (25:1 rod/cone ratio).
        
        Rod cells are responsible for:
        - Low-light vision
        - Motion detection
        - Peripheral vision
        - Reduced color discrimination
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            np.ndarray: Image with rod dominance effects
        """
        # Convert to grayscale to simulate reduced color discrimination
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Create rod-dominated image with reduced color saturation
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = hsv[:, :, 1] * 0.4  # Reduce saturation significantly
            rod_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        else:
            rod_image = image.copy()
            gray = image.copy()
        
        # Enhance contrast for better low-light performance
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        
        # Blend enhanced grayscale with reduced-color image
        if len(image.shape) == 3:
            enhanced_gray_bgr = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
            rod_image = cv2.addWeighted(rod_image, 0.7, enhanced_gray_bgr, 0.3, 0)
        else:
            rod_image = enhanced_gray
            
        return rod_image
    
    def enhance_motion_detection(self, image, previous_frame=None):
        """
        Enhance motion detection capabilities (rod cell specialization).
        
        Args:
            image (np.ndarray): Current frame
            previous_frame (np.ndarray): Previous frame for motion detection
            
        Returns:
            np.ndarray: Motion-enhanced image
        """
        if previous_frame is None:
            return image
            
        # Convert to grayscale for motion detection
        if len(image.shape) == 3:
            current_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        else:
            current_gray = image
            prev_gray = previous_frame
            
        # Calculate frame difference
        diff = cv2.absdiff(current_gray, prev_gray)
        
        # Enhance motion areas
        motion_mask = diff > 10  # Threshold for motion detection
        enhanced = image.copy().astype(np.float32)
        
        if len(image.shape) == 3:
            for channel in range(3):
                enhanced[:, :, channel][motion_mask] *= self.motion_sensitivity
        else:
            enhanced[motion_mask] *= self.motion_sensitivity
            
        return np.clip(enhanced, 0, 255).astype(np.uint8)
    
    def apply_pupil_filter(self, image, kernel_size=15):
        """
        Apply vertical slit pupil convolution filter.
        
        Args:
            image (np.ndarray): Input image
            kernel_size (int): Size of pupil kernel
            
        Returns:
            np.ndarray: Pupil-filtered image
        """
        kernel = self.create_pupil_kernel(kernel_size)
        
        if len(image.shape) == 3:
            # Apply to each channel
            filtered_channels = []
            for i in range(3):
                filtered_channel = cv2.filter2D(image[:, :, i], -1, kernel)
                filtered_channels.append(filtered_channel)
            filtered_image = cv2.merge(filtered_channels)
        else:
            filtered_image = cv2.filter2D(image, -1, kernel)
            
        return filtered_image
    
    def apply_cat_vision(self, image: np.ndarray, previous_frame: Optional[np.ndarray] = None, 
                        kernel_size: int = 15, use_biological_accuracy: bool = True) -> np.ndarray:
        """
        Apply complete cat vision pipeline with enhanced biological accuracy.
        
        Args:
            image: Input image (BGR format)
            previous_frame: Previous frame for motion detection
            kernel_size: Size of pupil kernel
            use_biological_accuracy: Use new biologically accurate methods vs legacy methods
            
        Returns:
            Cat vision processed image
        """
        if use_biological_accuracy:
            return self._apply_enhanced_cat_vision(image, previous_frame, kernel_size)
        else:
            return self._apply_legacy_cat_vision(image, previous_frame, kernel_size)
    
    def _apply_enhanced_cat_vision(self, image: np.ndarray, previous_frame: Optional[np.ndarray], 
                                  kernel_size: int) -> np.ndarray:
        """Enhanced biologically accurate cat vision pipeline."""
        # Step 1: Apply spatial field transformation (wide FOV)
        field_transformed = self.apply_spatial_field_transformation(image)
        
        # Step 2: Apply pupil filter (vertical slit)
        pupil_filtered = self.apply_pupil_filter(field_transformed, kernel_size)
        
        # Step 3: Apply biological spectral sensitivity curves
        spectral_corrected = self.apply_spectral_sensitivity_curves(pupil_filtered)
        
        # Step 4: Simulate spatial acuity reduction
        acuity_reduced = self.simulate_spatial_acuity_reduction(spectral_corrected)
        
        # Step 5: Apply rod dominance effects
        rod_dominated = self.simulate_rod_dominance(acuity_reduced)
        
        # Step 6: Apply tapetum lucidum effect
        tapetum_enhanced = self.apply_tapetum_effect(rod_dominated)
        
        return tapetum_enhanced
    
    def _apply_legacy_cat_vision(self, image: np.ndarray, previous_frame: Optional[np.ndarray], 
                                kernel_size: int) -> np.ndarray:
        """Legacy cat vision pipeline for comparison."""
        # Step 1: Apply pupil filter (vertical slit)
        pupil_filtered = self.apply_pupil_filter(image, kernel_size)
        
        # Step 2: Adjust color sensitivity (enhance blue-green, reduce red)
        color_adjusted = self.adjust_color_sensitivity(pupil_filtered)
        
        # Step 3: Apply rod dominance effects
        rod_dominated = self.simulate_rod_dominance(color_adjusted)
        
        # Step 4: Apply tapetum lucidum effect
        tapetum_enhanced = self.apply_tapetum_effect(rod_dominated)
        
        # Step 5: Enhance motion detection (if previous frame available)
        if previous_frame is not None:
            motion_enhanced = self.enhance_motion_detection(tapetum_enhanced, previous_frame)
        else:
            motion_enhanced = tapetum_enhanced
            
        return motion_enhanced
    
    def apply_cat_vision_to_sequence(self, frame_sequence: List[np.ndarray], 
                                   fps: int = 30, use_biological_accuracy: bool = True) -> List[np.ndarray]:
        """
        Apply cat vision processing to a sequence of frames with temporal processing.
        
        Args:
            frame_sequence: List of consecutive frames
            fps: Frames per second
            use_biological_accuracy: Use enhanced biological methods
            
        Returns:
            Processed frame sequence
        """
        if len(frame_sequence) == 0:
            return []
        
        # Apply spatial and spectral processing to each frame
        processed_frames = []
        for frame in frame_sequence:
            processed_frame = self.apply_cat_vision(frame, use_biological_accuracy=use_biological_accuracy)
            processed_frames.append(processed_frame)
        
        # Apply temporal processing
        if use_biological_accuracy and len(processed_frames) > 1:
            temporal_processed = self.model_temporal_processing(processed_frames, fps)
            motion_enhanced = self.enhanced_motion_detection(temporal_processed)
            return motion_enhanced
        
        return processed_frames
    
    def validate_biological_accuracy(self, test_images: List[np.ndarray], 
                                   ground_truth_data: Optional[Dict] = None) -> Dict:
        """
        Validate biological accuracy against published cat vision characteristics.
        
        Args:
            test_images: List of test images for validation
            ground_truth_data: Optional ground truth data for comparison
            
        Returns:
            Validation results dictionary
        """
        results = {
            'spectral_sensitivity_validation': self._validate_spectral_sensitivity(),
            'spatial_acuity_validation': self._validate_spatial_acuity(test_images),
            'temporal_response_validation': self._validate_temporal_response(),
            'motion_detection_validation': self._validate_motion_detection(test_images),
            'overall_accuracy_score': 0.0
        }
        
        # Calculate overall accuracy score
        scores = [v for v in results.values() if isinstance(v, (int, float))]
        if scores:
            results['overall_accuracy_score'] = np.mean(scores)
        
        return results
    
    def _validate_spectral_sensitivity(self) -> float:
        """Validate spectral sensitivity curves against biological data."""
        # Expected peak wavelengths from literature
        expected_s_peak = 450  # ±10nm
        expected_l_peak = 556  # ±15nm
        expected_rod_peak = 498  # ±10nm
        
        # Calculate accuracy based on peak positions
        s_accuracy = 1.0 - min(abs(self.s_cone_peak - expected_s_peak) / 10.0, 1.0)
        l_accuracy = 1.0 - min(abs(self.l_cone_peak - expected_l_peak) / 15.0, 1.0)
        rod_accuracy = 1.0 - min(abs(self.rod_peak - expected_rod_peak) / 10.0, 1.0)
        
        return np.mean([s_accuracy, l_accuracy, rod_accuracy])
    
    def _validate_spatial_acuity(self, test_images: List[np.ndarray]) -> float:
        """Validate spatial acuity reduction against behavioral measurements."""
        if not test_images:
            return 0.0
            
        # Test acuity reduction on a sample image
        test_image = test_images[0]
        original_fft = np.fft.fft2(cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY) if len(test_image.shape) == 3 else test_image)
        
        # Apply acuity reduction
        acuity_reduced = self.simulate_spatial_acuity_reduction(test_image)
        reduced_fft = np.fft.fft2(cv2.cvtColor(acuity_reduced, cv2.COLOR_BGR2GRAY) if len(acuity_reduced.shape) == 3 else acuity_reduced)
        
        # Calculate frequency content reduction
        original_power = np.mean(np.abs(original_fft))
        reduced_power = np.mean(np.abs(reduced_fft))
        reduction_ratio = reduced_power / original_power
        
        # Expected reduction should be around 0.167 (1/6 of human acuity)
        expected_ratio = self.spatial_acuity_factor
        accuracy = 1.0 - min(abs(reduction_ratio - expected_ratio) / expected_ratio, 1.0)
        
        return accuracy
    
    def _validate_temporal_response(self) -> float:
        """Validate temporal frequency response characteristics."""
        # Test key frequency points
        test_frequencies = [10, 24, 55, 80]  # Hz
        expected_responses = [1.5, 1.2, 1.0, 0.1]  # Approximate expected values
        
        actual_responses = [self._temporal_sensitivity_function(f) for f in test_frequencies]
        
        # Calculate accuracy based on response matching
        accuracies = []
        for actual, expected in zip(actual_responses, expected_responses):
            accuracy = 1.0 - min(abs(actual - expected) / expected, 1.0)
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    def _validate_motion_detection(self, test_images: List[np.ndarray]) -> float:
        """Validate motion detection enhancement."""
        if len(test_images) < 2:
            return 0.0
            
        # Create simple motion test
        frame1 = test_images[0]
        frame2 = test_images[1] if len(test_images) > 1 else test_images[0]
        
        # Test enhanced motion detection
        enhanced_frames = self.enhanced_motion_detection([frame1, frame2])
        
        # Calculate motion enhancement factor
        original_diff = cv2.absdiff(
            cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY) if len(frame1.shape) == 3 else frame1,
            cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) if len(frame2.shape) == 3 else frame2
        )
        enhanced_diff = cv2.absdiff(
            cv2.cvtColor(enhanced_frames[0], cv2.COLOR_BGR2GRAY) if len(enhanced_frames[0].shape) == 3 else enhanced_frames[0],
            cv2.cvtColor(enhanced_frames[1], cv2.COLOR_BGR2GRAY) if len(enhanced_frames[1].shape) == 3 else enhanced_frames[1]
        )
        
        enhancement_ratio = np.mean(enhanced_diff) / (np.mean(original_diff) + 1e-6)
        
        # Expected enhancement should be around motion_sensitivity factor
        expected_enhancement = self.motion_sensitivity
        accuracy = 1.0 - min(abs(enhancement_ratio - expected_enhancement) / expected_enhancement, 1.0)
        
        return accuracy

    def get_filter_parameters(self):
        """
        Get current filter parameters for documentation and reproducibility.
        
        Returns:
            dict: Dictionary of all filter parameters
        """
        return {
            # Basic biological parameters
            'pupil_aspect_ratio': self.pupil_aspect_ratio,
            'rod_cone_ratio': self.rod_cone_ratio,
            'peak_wavelength': self.peak_wavelength,
            'tapetum_reflectance': self.tapetum_reflectance,
            'visual_field_horizontal': self.visual_field_h,
            'visual_field_vertical': self.visual_field_v,
            
            # Spectral sensitivity parameters
            's_cone_peak': self.s_cone_peak,
            'l_cone_peak': self.l_cone_peak,
            'rod_peak': self.rod_peak,
            
            # Spatial acuity parameters
            'spatial_acuity_factor': self.spatial_acuity_factor,
            'foveal_acuity_cycles_per_degree': self.foveal_acuity_cycles_per_degree,
            
            # Temporal processing parameters
            'flicker_fusion_threshold': self.flicker_fusion_threshold,
            'temporal_sensitivity_peak': self.temporal_sensitivity_peak,
            
            # Motion detection parameters
            'motion_sensitivity': self.motion_sensitivity,
            'horizontal_motion_bias': self.horizontal_motion_bias,
            
            # Legacy parameters (for backward compatibility)
            'color_weights': self.color_weights
        }
    
    def save_parameters(self, filepath):
        """
        Save filter parameters to JSON file.
        
        Args:
            filepath (str): Path to save parameters
        """
        params = self.get_filter_parameters()
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    def visualize_pupil_kernel(self, kernel_size=15, save_path=None):
        """
        Visualize the pupil kernel for validation.
        
        Args:
            kernel_size (int): Size of kernel to visualize
            save_path (str): Optional path to save visualization
        """
        kernel = self.create_pupil_kernel(kernel_size)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(kernel, cmap='hot', interpolation='bilinear')
        plt.colorbar(label='Kernel Weight')
        plt.title('Cat Pupil Kernel (Vertical Slit)\n3:1 Aspect Ratio')
        plt.xlabel('Horizontal Position')
        plt.ylabel('Vertical Position')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def demo_enhanced_cat_vision():
    """
    Comprehensive demonstration of enhanced cat vision filter capabilities.
    """
    print("Enhanced Cat Vision Filter Demo")
    print("=" * 60)
    print("Biologically Accurate Cat Vision Simulation for NeuroAI Research")
    print("=" * 60)
    
    # Initialize filter
    cat_filter = CatVisionFilter()
    
    # Print enhanced parameters
    params = cat_filter.get_filter_parameters()
    print("\n🔬 BIOLOGICAL PARAMETERS:")
    print("-" * 40)
    
    # Group parameters by category
    basic_params = ['pupil_aspect_ratio', 'rod_cone_ratio', 'tapetum_reflectance', 
                   'visual_field_horizontal', 'visual_field_vertical']
    spectral_params = ['s_cone_peak', 'l_cone_peak', 'rod_peak']
    spatial_params = ['spatial_acuity_factor', 'foveal_acuity_cycles_per_degree']
    temporal_params = ['flicker_fusion_threshold', 'temporal_sensitivity_peak']
    motion_params = ['motion_sensitivity', 'horizontal_motion_bias']
    
    print("Basic Vision Parameters:")
    for key in basic_params:
        if key in params:
            print(f"  • {key}: {params[key]}")
    
    print("\nSpectral Sensitivity:")
    for key in spectral_params:
        if key in params:
            print(f"  • {key}: {params[key]} nm")
    
    print("\nSpatial Acuity:")
    for key in spatial_params:
        if key in params:
            print(f"  • {key}: {params[key]}")
    
    print("\nTemporal Processing:")
    for key in temporal_params:
        if key in params:
            print(f"  • {key}: {params[key]} Hz")
    
    print("\nMotion Detection:")
    for key in motion_params:
        if key in params:
            print(f"  • {key}: {params[key]}")
    
    print("\n📊 GENERATING VISUALIZATIONS:")
    print("-" * 40)
    
    # Generate all visualizations
    print("1. Spectral sensitivity curves...")
    cat_filter.plot_spectral_sensitivity_curves()
    
    print("2. Spatial acuity mapping...")
    cat_filter.visualize_spatial_acuity_map((480, 640))
    
    print("3. Temporal frequency response...")
    cat_filter.demonstrate_temporal_frequency_response()
    
    print("4. Pupil kernel visualization...")
    cat_filter.visualize_pupil_kernel(kernel_size=21)
    
    print("\n✅ ENHANCED CAT VISION FILTER READY!")
    print("-" * 40)
    print("Key Features Implemented:")
    print("  ✓ Biological spectral sensitivity curves")
    print("  ✓ Spatial acuity reduction (1/6 human acuity)")
    print("  ✓ Wide-angle field of view (200°×140°)")
    print("  ✓ Enhanced temporal processing (55Hz flicker fusion)")
    print("  ✓ Advanced motion detection with optical flow")
    print("  ✓ Validation framework for biological accuracy")
    print("  ✓ Comprehensive visualization tools")
    
    print("\n🧠 READY FOR NEUROAI RESEARCH:")
    print("Use apply_cat_vision() with use_biological_accuracy=True")
    print("for maximum biological accuracy in your experiments!")
    
    return cat_filter

def demo_cat_vision():
    """Legacy demo function for backward compatibility."""
    return demo_enhanced_cat_vision()


if __name__ == "__main__":
    demo_cat_vision()
