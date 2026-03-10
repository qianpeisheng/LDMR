"""
Enhanced Memory Bank for 3D Point Cloud Incremental Learning

Memory bank implementation that stores 3D object metadata and caches extracted 
point clouds to prevent catastrophic forgetting in incremental learning scenarios.

IMPORTANT DESIGN NOTES:
1. ScanNet dataset is ALWAYS available during memory bank operations (training)
2. When cache evicts, we re-extract from ScanNet dataset, not from disk files  
3. Saved .bin/.json files are FOR DEBUGGING AND VISUALIZATION ONLY
4. Each experiment gets its own saved memory bank files for debugging
5. The memory bank keeps point clouds in LRU cache, re-extracting as needed
"""

import numpy as np
import random
import copy
import os
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod


class SelectionStrategy(ABC):
    """Abstract base class for exemplar selection strategies."""
    
    @abstractmethod
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """Select exemplar indices from objects."""
        pass


class RandomSelection(SelectionStrategy):
    """Random selection strategy."""
    
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """Randomly select exemplars."""
        if num_to_select >= len(objects):
            return list(range(len(objects)))
        return random.sample(range(len(objects)), num_to_select)


class ConfidenceSelection(SelectionStrategy):
    """Confidence-based selection strategy."""
    
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """Select exemplars with highest confidence scores."""
        confidences = kwargs.get('confidences', None)
        if confidences is None:
            # Fallback to random if no confidences provided
            return RandomSelection().select(objects, num_to_select)
        
        sorted_indices = sorted(range(len(objects)), 
                              key=lambda i: confidences[i], reverse=True)
        return sorted_indices[:num_to_select]


# Placeholder strategies for future implementation
class HerdingSelection(SelectionStrategy):
    """Herding-based selection strategy (placeholder)."""
    
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """TODO: Implement herding strategy - select closest to class mean."""
        print("⚠️  Herding selection not implemented, falling back to random")
        return RandomSelection().select(objects, num_to_select)


class DiversitySelection(SelectionStrategy):
    """Diversity-aware selection strategy (placeholder)."""
    
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """TODO: Implement diversity-aware strategy - ensure diverse exemplars."""
        print("⚠️  Diversity selection not implemented, falling back to random")
        return RandomSelection().select(objects, num_to_select)


class UncertaintySelection(SelectionStrategy):
    """Uncertainty-based selection strategy (placeholder)."""
    
    def select(self, objects: List[Dict], num_to_select: int, **kwargs) -> List[int]:
        """TODO: Implement uncertainty-based strategy - select uncertain samples."""
        print("⚠️  Uncertainty selection not implemented, falling back to random")
        return RandomSelection().select(objects, num_to_select)


class MemoryBank:
    """Enhanced memory bank for 3D point cloud incremental learning."""
    
    SELECTION_STRATEGIES = {
        'random': RandomSelection,
        'confidence': ConfidenceSelection,
        'herding': HerdingSelection,
        'diversity': DiversitySelection,
        'uncertainty': UncertaintySelection
    }
    
    def __init__(self, 
                 exemplars_per_class: int = 20,
                 selection_strategy: str = 'random',
                 max_total_exemplars: int = 1000,
                 cache_extracted_objects: bool = True,
                 max_cache_size: int = 200,  # MB
                 work_dir: str = None):
        """
        Args:
            exemplars_per_class (int): Number of exemplars to store per class
            selection_strategy (str): Strategy for selecting exemplars
            max_total_exemplars (int): Maximum total exemplars to prevent memory overflow
            cache_extracted_objects (bool): Whether to cache extracted point clouds
            max_cache_size (int): Maximum cache size in MB
            work_dir (str): Work directory for fallback loading of .bin files
        """
        self.exemplars_per_class = exemplars_per_class
        self.selection_strategy_name = selection_strategy
        self.max_total_exemplars = max_total_exemplars
        self.cache_extracted_objects = cache_extracted_objects
        self.max_cache_size = max_cache_size * 1024 * 1024  # Convert MB to bytes
        self.work_dir = work_dir  # Store for fallback loading
        
        # Initialize selection strategy
        if selection_strategy in self.SELECTION_STRATEGIES:
            self.selection_strategy = self.SELECTION_STRATEGIES[selection_strategy]()
        else:
            print(f"⚠️  Unknown selection strategy: {selection_strategy}, using random")
            self.selection_strategy = RandomSelection()
        
        # Storage: class_id -> list of exemplar metadata
        self.exemplars = {}
        self.exemplar_count = 0
        
        # Point cloud cache: (scene_id, object_idx) -> cached point cloud data
        self.point_cloud_cache = {}
        self.cache_size_bytes = 0
        
        # Statistics
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Track reduction history and active exemplars
        self.reduction_history = []
        self.removed_exemplars = {}  # Track removed exemplars per class
        
        print(f"Enhanced Memory Bank initialized:")
        print(f"  Exemplars per class: {exemplars_per_class}")
        print(f"  Selection strategy: {selection_strategy}")
        print(f"  Max total exemplars: {max_total_exemplars}")
        print(f"  Point cloud caching: {cache_extracted_objects}")
        print(f"  Max cache size: {max_cache_size}MB")
    
    @staticmethod
    def extract_object_points(scene_points: np.ndarray, bbox: np.ndarray, debug_info: str = "unknown") -> np.ndarray:
        """Extract points within a bounding box from scene points.
        
        CRITICAL FIX: Enhanced with comprehensive debugging and coordinate validation.
        
        Args:
            scene_points (np.ndarray): Full scene point cloud [N, 3+] 
            bbox (np.ndarray): Bounding box [x, y, z, dx, dy, dz, heading]
            debug_info (str): Debug identifier for logging
            
        Returns:
            np.ndarray: Object point cloud [M, 3+]
        """
        if len(scene_points) == 0:
            print(f"⚠️  extract_object_points({debug_info}): Empty scene points")
            return np.array([]).reshape(0, scene_points.shape[1] if len(scene_points.shape) > 1 else 3)
        
        # Validate bbox format
        if bbox is None or len(bbox) < 6:
            print(f"❌ extract_object_points({debug_info}): Invalid bbox format: {bbox}")
            return np.array([]).reshape(0, scene_points.shape[1])
        
        # Extract bounding box parameters
        center = bbox[:3].astype(np.float32)
        size = bbox[3:6].astype(np.float32)
        heading = float(bbox[6]) if len(bbox) > 6 else 0.0
        
        # Debug logging (only if extraction fails)
        debug_verbose = False  # Set to True for detailed debugging
        
        # Validate bbox parameters
        if np.any(size <= 0):
            print(f"❌ extract_object_points({debug_info}): Invalid bbox size: {size}")
            return np.array([]).reshape(0, scene_points.shape[1])
        
        if np.any(np.isnan(center)) or np.any(np.isnan(size)):
            print(f"❌ extract_object_points({debug_info}): NaN in bbox parameters")
            return np.array([]).reshape(0, scene_points.shape[1])
        
        # Get points in bbox coordinate system
        points_xyz = scene_points[:, :3].astype(np.float32)
        
        # Translate to bbox center
        points_relative = points_xyz - center
        
        # Rotate if needed (TR3D typically uses heading angle around Z-axis)
        if abs(heading) > 1e-6:
            cos_h = np.cos(heading)
            sin_h = np.sin(heading)
            rotation_matrix = np.array([
                [cos_h, -sin_h, 0],
                [sin_h, cos_h, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            points_relative = points_relative @ rotation_matrix.T
            print(f"  Applied rotation: {np.degrees(heading):.2f}°")
        
        # Check if points are within bounding box
        half_size = size / 2
        mask = (
            (np.abs(points_relative[:, 0]) <= half_size[0]) &
            (np.abs(points_relative[:, 1]) <= half_size[1]) &
            (np.abs(points_relative[:, 2]) <= half_size[2])
        )
        
        points_in_bbox_count = mask.sum()
        
        if points_in_bbox_count == 0:
            if debug_verbose:
                print(f"⚠️  extract_object_points({debug_info}): No points found in bounding box!")
                print(f"  Scene points shape: {scene_points.shape}")
                print(f"  Scene bounds: [{points_xyz.min(axis=0)}, {points_xyz.max(axis=0)}]")
                print(f"  Bbox center: {center}")
                print(f"  Bbox size: {size}")
                print(f"  Bbox extent: center={center} ± {half_size}")
                # Check if bbox is completely outside scene bounds
                scene_min, scene_max = points_xyz.min(axis=0), points_xyz.max(axis=0)
                bbox_min, bbox_max = center - half_size, center + half_size
                if (np.all(bbox_max < scene_min) or np.all(bbox_min > scene_max)):
                    print(f"  ❌ CRITICAL: Bounding box is completely outside scene bounds!")
            return np.array([]).reshape(0, scene_points.shape[1])
        
        object_points = scene_points[mask]
        
        # Normalize to object coordinate system (center at origin) 
        # CRITICAL: Keep original scene coordinates for proper insertion
        if len(object_points) > 0:
            object_points = object_points.copy()
            # Store relative coordinates (centered at bbox center)
            object_points[:, :3] = points_relative[mask]
            if debug_verbose:
                print(f"✅ extract_object_points({debug_info}): Successfully extracted {len(object_points)} points")
        
        return object_points
    
    def get_cached_object_points(self, scene_id: str, object_idx: int, 
                                scene_points: np.ndarray = None, 
                                bbox: np.ndarray = None,
                                class_name: str = "unknown") -> Optional[np.ndarray]:
        """Get cached object points or extract and cache them.
        
        Args:
            scene_id (str): Scene identifier
            object_idx (int): Object index within scene
            scene_points (np.ndarray, optional): Full scene points for extraction
            bbox (np.ndarray, optional): Object bounding box for extraction
            class_name (str): Class name for debugging purposes
            
        Returns:
            np.ndarray or None: Cached object points
        """
        cache_key = (scene_id, object_idx)
        
        # Check cache first
        if cache_key in self.point_cloud_cache:
            self.cache_hits += 1
            return self.point_cloud_cache[cache_key]
        
        # Cache miss - need to extract
        self.cache_misses += 1
        
        if scene_points is None or bbox is None:
            print(f"⚠️  Cannot extract object {class_name}({scene_id}:{object_idx}): missing scene_points or bbox")
            return None
        
        # Extract object points with debug info
        debug_info = f"{class_name}:{scene_id}:{object_idx}"
        object_points = self.extract_object_points(scene_points, bbox, debug_info)
        
        if not self.cache_extracted_objects:
            return object_points
        
        # Cache the extracted points
        self._add_to_cache(cache_key, object_points)
        return object_points
    
    def _add_to_cache(self, cache_key: Tuple[str, int], object_points: np.ndarray):
        """Add object points to cache with memory management."""
        # Estimate memory usage (approximate)
        obj_size = object_points.nbytes
        
        # Check if we need to free cache space
        while (self.cache_size_bytes + obj_size > self.max_cache_size and 
               len(self.point_cloud_cache) > 0):
            # Simple FIFO cache eviction
            oldest_key = next(iter(self.point_cloud_cache))
            removed_obj = self.point_cloud_cache.pop(oldest_key)
            self.cache_size_bytes -= removed_obj.nbytes
            print(f"Cache evicted: {oldest_key}")
        
        # Add to cache
        self.point_cloud_cache[cache_key] = object_points
        self.cache_size_bytes += obj_size
    
    def add_exemplars(self, class_id: int, objects: List[Dict], 
                     confidences: Optional[List[float]] = None,
                     scene_points_dict: Optional[Dict[str, np.ndarray]] = None,
                     debug_save_dir: Optional[str] = None,
                     stage_id: Optional[int] = None):
        """Add exemplars for a specific class using metadata storage.
        
        Args:
            class_id (int): Class index to add exemplars for
            objects (List[Dict]): List of object metadata 
                Each dict should contain: scene_id, object_idx, bbox, class_id
            confidences (List[float], optional): Confidence scores for selection
            scene_points_dict (Dict[str, np.ndarray], optional): Scene points for caching
            debug_save_dir (str, optional): Directory to save debug files
            stage_id (int, optional): Current stage ID for debug file organization
        """
        # EDGE CASE 1: Empty objects list
        if not objects:
            print(f"⚠️  EDGE CASE: No objects provided for class {class_id}")
            # Still create empty entry to track that we attempted to add this class
            if class_id not in self.exemplars:
                self.exemplars[class_id] = []
            return
        
        # EDGE CASE 2: Insufficient objects
        available_objects = len(objects)
        requested_exemplars = self.exemplars_per_class
        num_to_select = min(available_objects, requested_exemplars)
        
        if available_objects < requested_exemplars:
            print(f"⚠️  EDGE CASE: Insufficient objects for class {class_id}")
            print(f"    Requested: {requested_exemplars}, Available: {available_objects}")
            print(f"    Will store all {available_objects} available objects")
        
        # Select exemplars using strategy
        selected_indices = self.selection_strategy.select(
            objects, num_to_select, confidences=confidences
        )
        
        # Store selected exemplars metadata
        selected_exemplars = []
        for idx in selected_indices:
            obj = objects[idx]
            exemplar = {
                'scene_id': obj['scene_id'],
                'object_idx': obj.get('object_idx', idx),
                'bbox': obj['bbox'],
                'class_id': class_id,
                'confidence': obj.get('confidence', 1.0),
                'nyu40_id': obj.get('nyu40_id', None),
                'point_count': 0  # Initialize to 0, will be updated if extraction succeeds
            }
            selected_exemplars.append(exemplar)
            
            # Cache object points if scene data available (CRITICAL FIX: Validate extraction)
            if (scene_points_dict is not None and 
                obj['scene_id'] in scene_points_dict):
                scene_points = scene_points_dict[obj['scene_id']]
                class_name = f"class_{class_id}"  # Could be improved with actual class names
                
                # Extract and validate points
                extracted_points = self.get_cached_object_points(
                    obj['scene_id'], exemplar['object_idx'], 
                    scene_points, obj['bbox'], class_name
                )
                
                # VALIDATION: Check if extraction was successful
                if extracted_points is None or len(extracted_points) == 0:
                    # Still add to exemplar list but mark as failed for debugging
                    exemplar['extraction_failed'] = True
                    exemplar['point_count'] = 0
                else:
                    exemplar['extraction_failed'] = False
                    exemplar['point_count'] = len(extracted_points)
        
        if class_id not in self.exemplars:
            self.exemplars[class_id] = []
        
        # Replace existing exemplars for this class
        self.exemplars[class_id] = selected_exemplars
        self._update_exemplar_count()
        
        # EDGE CASE 3: Check if we're approaching or exceeding max total exemplars
        if self.exemplar_count > self.max_total_exemplars * 0.8:
            print(f"⚠️  EDGE CASE: Memory bank approaching limit")
            print(f"    Current: {self.exemplar_count}/{self.max_total_exemplars} ({self.exemplar_count/self.max_total_exemplars*100:.1f}%)")
            if self.exemplar_count > self.max_total_exemplars:
                print(f"🚨 EDGE CASE: Memory bank overflow! Triggering reduction...")
                self._reduce_exemplars()
        
        # VALIDATION SUMMARY: Report extraction success/failure rates
        failed_extractions = sum(1 for ex in selected_exemplars if ex.get('extraction_failed', False))
        successful_extractions = len(selected_exemplars) - failed_extractions
        success_rate = (successful_extractions / len(selected_exemplars)) * 100 if selected_exemplars else 0
        
        print(f"📊 EXEMPLAR EXTRACTION SUMMARY for class {class_id}:")
        print(f"  Total selected: {len(selected_exemplars)}")
        print(f"  Successful extractions: {successful_extractions}")
        print(f"  Failed extractions: {failed_extractions}")
        print(f"  Success rate: {success_rate:.1f}%")
        
        if success_rate < 50:
            print(f"🚨 WARNING: Low extraction success rate ({success_rate:.1f}%) for class {class_id}")
            print("    This indicates potential coordinate system or bbox issues!")
        
        # Save debug files if requested
        if debug_save_dir is not None and stage_id is not None:
            self._save_exemplar_debug_files(class_id, selected_exemplars, debug_save_dir, stage_id)
    
    def get_exemplars(self, class_ids: List[int]) -> List[Dict]:
        """Get exemplar metadata for specified classes."""
        exemplars = []
        
        for class_id in class_ids:
            if class_id in self.exemplars:
                exemplars.extend(self.exemplars[class_id])
        
        return exemplars
    
    
    def get_exemplar_points(self, exemplar: Dict, 
                           scene_points_dict: Optional[Dict[str, np.ndarray]] = None,
                           dataset_ref=None) -> Optional[np.ndarray]:
        """Get point cloud data for a specific exemplar.
        
        If not in cache, re-extract from ScanNet dataset (always available for memory bank).
        
        Args:
            exemplar (Dict): Exemplar metadata
            scene_points_dict (Dict, optional): Pre-loaded scene points
            dataset_ref: Reference to dataset for re-extraction
            
        Returns:
            np.ndarray or None: Object point cloud
        """
        scene_id = exemplar['scene_id']
        object_idx = exemplar['object_idx']
        bbox = exemplar['bbox']
        
        # Try to get from cache first
        class_name = f"class_{exemplar.get('class_id', 'unknown')}"
        cached_points = self.get_cached_object_points(scene_id, object_idx, class_name=class_name)
        if cached_points is not None:
            return cached_points
        
        # Re-extract from ScanNet (always available for memory bank operations)
        scene_points = None
        
        # First check if we have pre-loaded scene points
        if scene_points_dict and scene_id in scene_points_dict:
            scene_points = scene_points_dict[scene_id]
        # Otherwise load from dataset (ScanNet is always available)
        elif dataset_ref and hasattr(dataset_ref, '_load_scene_points'):
            scene_points = dataset_ref._load_scene_points(scene_id)
            
        if scene_points is None:
            print(f"⚠️  Cannot get points for exemplar {class_name}({scene_id}:{object_idx}) - unable to load scene")
            return None
        
        # Extract and cache the points
        extracted_points = self.get_cached_object_points(
            scene_id, object_idx, 
            scene_points, bbox, class_name
        )
        
        return extracted_points
    
    def get_all_exemplars(self) -> List[Dict]:
        """Get all stored exemplar metadata."""
        all_exemplars = []
        for class_exemplars in self.exemplars.values():
            all_exemplars.extend(class_exemplars)
        return all_exemplars
    
    def get_class_exemplar_count(self, class_id: int) -> int:
        """Get number of exemplars stored for a specific class."""
        return len(self.exemplars.get(class_id, []))
    
    def get_total_exemplar_count(self) -> int:
        """Get total number of stored exemplars."""
        return self.exemplar_count
    
    def get_stored_classes(self) -> List[int]:
        """Get list of classes that have stored exemplars."""
        return list(self.exemplars.keys())
    
    def clear_class_exemplars(self, class_id: int):
        """Clear exemplars for a specific class."""
        if class_id in self.exemplars:
            del self.exemplars[class_id]
            self._update_exemplar_count()
            print(f"Cleared exemplars for class {class_id}")
    
    def clear_all_exemplars(self):
        """Clear all stored exemplars and cache."""
        self.exemplars.clear()
        self.exemplar_count = 0
        self.point_cloud_cache.clear()
        self.cache_size_bytes = 0
        print("Cleared all exemplars and cache from memory bank")
    
    def _update_exemplar_count(self):
        """Update total exemplar count."""
        self.exemplar_count = sum(len(exemplars) for exemplars in self.exemplars.values())
    
    def _reduce_exemplars(self):
        """Reduce number of exemplars if we exceed the total limit.
        
        Enhanced reduction strategy:
        1. Maintain minimum exemplars per class (at least 2)
        2. Prioritize recent classes (higher class IDs)
        3. Log reduction decisions for debugging
        """
        if self.exemplar_count <= self.max_total_exemplars:
            return
        
        print(f"📉 Memory Bank Reduction Starting")
        print(f"   Current: {self.exemplar_count} exemplars")
        print(f"   Target: {self.max_total_exemplars} exemplars")
        print(f"   Need to remove: {self.exemplar_count - self.max_total_exemplars} exemplars")
        
        # Calculate reduction strategy
        min_per_class = 2  # Minimum exemplars to keep per class
        reduction_log = []
        
        # First pass: ensure minimum exemplars
        total_after_min = 0
        classes_at_min = []
        for class_id, exemplars in self.exemplars.items():
            if len(exemplars) <= min_per_class:
                total_after_min += len(exemplars)
                classes_at_min.append(class_id)
            else:
                total_after_min += min_per_class
        
        if total_after_min >= self.max_total_exemplars:
            # EDGE CASE: Even with minimum exemplars, we exceed limit
            print(f"🚨 CRITICAL: Cannot maintain {min_per_class} exemplars per class")
            print(f"   Minimum would require: {total_after_min}")
            print(f"   Available: {self.max_total_exemplars}")
            # Fall back to proportional reduction with min=1
            min_per_class = 1
        
        # Second pass: proportional reduction above minimum
        excess_to_distribute = self.max_total_exemplars - total_after_min
        
        # Sort classes by ID (prioritize keeping more recent classes)
        sorted_classes = sorted(self.exemplars.keys(), reverse=True)
        
        new_exemplars = {}
        remaining_quota = self.max_total_exemplars
        
        for class_id in sorted_classes:
            current_count = len(self.exemplars[class_id])
            
            if remaining_quota <= 0:
                # No more space
                new_count = 0
            elif class_id in classes_at_min:
                # Keep current count (already at or below minimum)
                new_count = current_count
            else:
                # Calculate proportional allocation
                proportion = current_count / self.exemplar_count
                allocated = max(min_per_class, int(self.max_total_exemplars * proportion))
                new_count = min(current_count, allocated, remaining_quota)
            
            if new_count > 0:
                new_exemplars[class_id] = self.exemplars[class_id][:new_count]
                remaining_quota -= new_count
                
            # Track removed exemplars
            if new_count < current_count:
                if class_id not in self.removed_exemplars:
                    self.removed_exemplars[class_id] = []
                # Store removed exemplars for tracking
                removed = self.exemplars[class_id][new_count:]
                self.removed_exemplars[class_id].extend(removed)
                
            # Log reduction decision
            if new_count != current_count:
                reduction_log.append(f"Class {class_id}: {current_count} → {new_count} (-{current_count - new_count})")
        
        # Track reduction event
        reduction_event = {
            'timestamp': time.time() if 'time' in dir() else None,
            'before_count': sum(len(exs) for exs in self.exemplars.values()),
            'after_count': sum(len(exs) for exs in new_exemplars.values()),
            'removed_count': sum(len(exs) for exs in self.exemplars.values()) - sum(len(exs) for exs in new_exemplars.values()),
            'classes_affected': len(reduction_log)
        }
        self.reduction_history.append(reduction_event)
        
        # Apply reduction
        self.exemplars = new_exemplars
        self._update_exemplar_count()
        
        # Report reduction results
        print(f"✅ Memory Bank Reduction Complete")
        print(f"   Final count: {self.exemplar_count}/{self.max_total_exemplars}")
        print(f"   Classes affected: {len(reduction_log)}")
        if reduction_log:
            print("   Reduction details:")
            for log_entry in reduction_log[:5]:  # Show first 5
                print(f"     - {log_entry}")
            if len(reduction_log) > 5:
                print(f"     ... and {len(reduction_log) - 5} more classes")
    
    def _save_exemplar_debug_files(self, class_id: int, exemplars: List[Dict], debug_save_dir: str, stage_id: int):
        """Save exemplar data to debug files for inspection.
        
        IMPORTANT: These saved files are FOR DEBUGGING AND VISUALIZATION ONLY.
        They are NOT used operationally - the memory bank uses ScanNet dataset 
        for re-extraction when cache misses occur.
        """
        import json
        
        # Use debug_save_dir directly since it's already stage-specific
        stage_dir = debug_save_dir
        os.makedirs(stage_dir, exist_ok=True)
        
        # Save individual exemplars (CRITICAL FIX: Save ALL exemplars, even empty ones)
        for i, exemplar in enumerate(exemplars):
            bin_filename = f"class_{class_id}_exemplar_{i}.bin"
            bin_filepath = os.path.join(stage_dir, bin_filename)
            
            # Try to get points from cache
            cache_key = (exemplar['scene_id'], exemplar['object_idx'])
            points = None
            extraction_status = "not_cached"
            
            if cache_key in self.point_cloud_cache:
                points = self.point_cloud_cache[cache_key]
                extraction_status = "cached"
                print(f"💾 Saving cached exemplar {class_id}:{i} -> {bin_filename} ({len(points)} points)")
            else:
                print(f"⚠️  Exemplar {class_id}:{i} not in cache - creating empty file for debugging")
                points = np.array([]).reshape(0, 6)  # Empty array with proper shape
                extraction_status = "cache_miss"
            
            # Always save .bin file (even if empty)
            points.astype(np.float32).tofile(bin_filepath)
            
            # Save metadata with extraction status
            meta_filename = f"class_{class_id}_exemplar_{i}_meta.json"
            meta_filepath = os.path.join(stage_dir, meta_filename)
            exemplar_meta = {
                'scene_id': exemplar['scene_id'],
                'object_idx': int(exemplar['object_idx']),  # Convert numpy.int64 to int
                'class_id': int(exemplar['class_id']),      # Convert numpy.int64 to int
                'bbox': exemplar['bbox'].tolist() if isinstance(exemplar['bbox'], np.ndarray) else exemplar['bbox'],
                'confidence': float(exemplar.get('confidence', 1.0)),  # Ensure float
                'nyu40_id': int(exemplar.get('nyu40_id')) if exemplar.get('nyu40_id') is not None else None,
                'point_count': len(points),
                'point_file': bin_filename,
                'extraction_status': extraction_status,  # Track why this exemplar succeeded/failed
                'cache_key': f"{cache_key[0]}:{cache_key[1]}"  # Debug info
            }
            with open(meta_filepath, 'w') as f:
                json.dump(exemplar_meta, f, indent=2)
        
        # Save unified stage summary file
        self._save_stage_summary(stage_dir, stage_id, class_id, exemplars)
    
    def _save_stage_summary(self, stage_dir: str, stage_id: int, class_id: int, exemplars: List[Dict]):
        """Save unified stage summary in a single JSON file."""
        import json
        import time
        
        summary_file = os.path.join(os.path.dirname(stage_dir), 'stage_summary.json')
        
        # Load existing summary or create new one
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
        else:
            summary = {
                'stage_id': stage_id,
                'classes': {},
                'total_exemplars': 0,
                'selection_strategy': self.selection_strategy_name
            }
        
        # Add this class to summary
        summary['classes'][str(class_id)] = {
            'exemplar_count': len(exemplars),
            'class_id': class_id,
            'files': [f"class_{class_id}_exemplar_{i}.bin" for i in range(len(exemplars))]
        }
        summary['total_exemplars'] = sum(cls_info['exemplar_count'] for cls_info in summary['classes'].values())
        
        # Save updated summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"💾 Debug files saved for class {class_id}: {len(exemplars)} exemplars in {stage_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory bank statistics with edge case detection."""
        # Identify edge cases
        empty_classes = [k for k, v in self.exemplars.items() if len(v) == 0]
        insufficient_classes = [k for k, v in self.exemplars.items() 
                               if 0 < len(v) < self.exemplars_per_class]
        at_limit_classes = [k for k, v in self.exemplars.items() 
                           if len(v) == self.exemplars_per_class]
        
        stats = {
            'total_exemplars': self.exemplar_count,
            'stored_classes': len(self.exemplars),
            'exemplars_per_class': {str(k): len(v) for k, v in self.exemplars.items()},
            'average_exemplars_per_class': self.exemplar_count / max(1, len(self.exemplars)),
            'memory_utilization': self.exemplar_count / self.max_total_exemplars * 100,
            'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
            'cache_entries': len(self.point_cloud_cache),
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100,
            'selection_strategy': self.selection_strategy_name,
            # Edge case statistics
            'edge_cases': {
                'empty_classes': empty_classes,
                'insufficient_classes': insufficient_classes,
                'classes_at_limit': at_limit_classes,
                'is_at_max_capacity': self.exemplar_count >= self.max_total_exemplars,
                'overflow_risk': self.exemplar_count > self.max_total_exemplars * 0.9,
            }
        }
        return stats
    
    def save_active_manifest(self, filepath: str, stage_id: Optional[int] = None):
        """Save manifest of currently active exemplars.
        
        Args:
            filepath: Path to save the manifest JSON file
            stage_id: Optional stage identifier for tracking
        """
        import json
        import time
        
        manifest = {
            'timestamp': time.strftime('%Y-%m-%d_%H:%M:%S'),
            'stage_id': stage_id,
            'total_exemplars': self.exemplar_count,
            'max_exemplars': self.max_total_exemplars,
            'memory_utilization': (self.exemplar_count / self.max_total_exemplars * 100),
            'active_exemplars': {},
            'removed_exemplars': {},
            'reduction_history': self.reduction_history,
            'statistics': {
                'cache_size_mb': self.cache_size_bytes / (1024 * 1024),
                'cache_entries': len(self.point_cloud_cache),
                'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses) * 100
            }
        }
        
        # Document active exemplars
        for class_id, exemplars in self.exemplars.items():
            manifest['active_exemplars'][str(class_id)] = []
            for i, exemplar in enumerate(exemplars):
                exemplar_info = {
                    'index': i,
                    'scene_id': exemplar['scene_id'],
                    'object_idx': exemplar['object_idx'],
                    'bbox': exemplar['bbox'].tolist() if hasattr(exemplar['bbox'], 'tolist') else exemplar['bbox'],
                    'file': f"class_{class_id}_exemplar_{i}.bin",
                    'active': True
                }
                manifest['active_exemplars'][str(class_id)].append(exemplar_info)
        
        # Document removed exemplars
        for class_id, removed in self.removed_exemplars.items():
            manifest['removed_exemplars'][str(class_id)] = []
            for exemplar in removed:
                removed_info = {
                    'scene_id': exemplar['scene_id'],
                    'object_idx': exemplar['object_idx'],
                    'active': False,
                    'removal_time': 'during_reduction'
                }
                manifest['removed_exemplars'][str(class_id)].append(removed_info)
        
        # Save manifest
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"📁 Active exemplar manifest saved: {filepath}")
        print(f"   Active: {self.exemplar_count} exemplars")
        print(f"   Removed: {sum(len(r) for r in self.removed_exemplars.values())} exemplars")
    
    def load_active_manifest(self, filepath: str):
        """Load active exemplar manifest for recovery or debugging.
        
        Args:
            filepath: Path to the manifest JSON file
            
        Returns:
            Loaded manifest dictionary
        """
        import json
        
        if not os.path.exists(filepath):
            print(f"⚠️  Manifest file not found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            manifest = json.load(f)
        
        print(f"📄 Loaded active exemplar manifest from: {filepath}")
        print(f"   Timestamp: {manifest['timestamp']}")
        print(f"   Stage: {manifest.get('stage_id', 'unknown')}")
        print(f"   Active exemplars: {manifest['total_exemplars']}")
        print(f"   Removed exemplars: {sum(len(r) for r in manifest['removed_exemplars'].values())}")
        
        return manifest
    
    def print_statistics(self):
        """Print comprehensive memory bank statistics with edge case reporting."""
        stats = self.get_statistics()
        print(f"📊 Enhanced Memory Bank Statistics:")
        print(f"  Total exemplars: {stats['total_exemplars']}/{self.max_total_exemplars} ({stats['memory_utilization']:.1f}%)")
        print(f"  Stored classes: {stats['stored_classes']}")
        print(f"  Average exemplars per class: {stats['average_exemplars_per_class']:.1f}")
        print(f"  Selection strategy: {stats['selection_strategy']}")
        print(f"  Cache: {stats['cache_size_mb']:.1f}MB ({stats['cache_entries']} entries, {stats['cache_hit_rate']:.1f}% hit rate)")
        
        # Report reduction history
        if self.reduction_history:
            print(f"  Reductions performed: {len(self.reduction_history)}")
            latest = self.reduction_history[-1]
            print(f"    Latest: {latest['before_count']} → {latest['after_count']} (-{latest['removed_count']})")
        
        # Report edge cases if any
        edge_cases = stats['edge_cases']
        if edge_cases['empty_classes'] or edge_cases['insufficient_classes'] or edge_cases['overflow_risk']:
            print(f"\n⚠️  Edge Cases Detected:")
            if edge_cases['empty_classes']:
                print(f"  Empty classes (0 exemplars): {edge_cases['empty_classes']}")
            if edge_cases['insufficient_classes']:
                print(f"  Insufficient exemplars: {edge_cases['insufficient_classes']}")
            if edge_cases['is_at_max_capacity']:
                print(f"  🚨 AT MAXIMUM CAPACITY")
            elif edge_cases['overflow_risk']:
                print(f"  ⚠️  Near maximum capacity (>90% full)")