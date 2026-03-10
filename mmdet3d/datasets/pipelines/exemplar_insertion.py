"""
In-scene exemplar insertion transform for incremental 3D object detection.

This transform inserts exemplar objects from previous stages into current stage scenes
to prevent catastrophic forgetting during incremental learning.
"""

import numpy as np
import copy
import random
from mmdet3d.core.points import get_points_type
from mmdet3d.core.bbox.structures import get_box_type
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class InsertExemplarObjects:
    """Insert exemplar objects from memory bank into training scenes.
    
    This transform implements in-scene exemplar insertion for incremental learning.
    Instead of treating exemplars as separate training samples, it inserts exemplar
    objects directly into current stage scenes to prevent catastrophic forgetting.
    
    Args:
        memory_bank: Memory bank containing exemplars from previous stages
        max_exemplars_per_scene (int): Maximum number of exemplars to insert per scene
        insertion_probability (float): Probability of inserting exemplars (0.0-1.0)
        collision_threshold (float): IoU threshold for collision detection
        placement_jitter (float): Random translation jitter for placement (meters)
        max_placement_attempts (int): Maximum attempts to find valid placement
        floor_offset (float): Vertical offset from floor for placement
    """
    
    def __init__(self,
                 memory_bank=None,
                 max_exemplars_per_scene=3,
                 insertion_probability=0.7,
                 collision_threshold=0.3,
                 placement_jitter=1.0,
                 max_placement_attempts=10,
                 floor_offset=0.0):
        self.memory_bank = memory_bank
        self.max_exemplars_per_scene = max_exemplars_per_scene
        self.insertion_probability = insertion_probability
        self.collision_threshold = collision_threshold
        self.placement_jitter = placement_jitter
        self.max_placement_attempts = max_placement_attempts
        self.floor_offset = floor_offset
    
    def __call__(self, results):
        """Insert exemplar objects into the scene.
        
        Args:
            results (dict): Scene data containing 'points', 'gt_bboxes_3d', 'gt_labels_3d'
            
        Returns:
            dict: Scene data with inserted exemplar objects
        """
        # Skip insertion if no memory bank or with certain probability
        if (self.memory_bank is None or 
            not hasattr(self.memory_bank, 'previous_classes') or
            random.random() > self.insertion_probability):
            return results
        
        # Get exemplars from memory bank
        previous_classes = getattr(self.memory_bank, 'previous_classes', [])
        if not previous_classes:
            return results
            
        # Sample exemplars for insertion
        exemplars_to_insert = self._sample_exemplars_for_insertion(previous_classes)
        if not exemplars_to_insert:
            return results
            
        # Insert exemplars into scene
        return self._insert_exemplars_into_scene(results, exemplars_to_insert)
    
    def _sample_exemplars_for_insertion(self, previous_classes):
        """Sample exemplars from memory bank for insertion."""
        if not hasattr(self.memory_bank, 'get_exemplars'):
            return []
            
        # Get all available exemplars
        available_exemplars = self.memory_bank.get_exemplars(previous_classes)
        if not available_exemplars:
            return []
        
        # Sample up to max_exemplars_per_scene
        num_to_sample = min(len(available_exemplars), self.max_exemplars_per_scene)
        if num_to_sample == 0:
            return []
            
        # Random sampling (could be enhanced with other strategies)
        sampled_exemplars = random.sample(available_exemplars, num_to_sample)
        return sampled_exemplars
    
    def _insert_exemplars_into_scene(self, results, exemplars):
        """Insert exemplar objects into the scene with collision detection."""
        points = results['points']
        gt_bboxes_3d = results['gt_bboxes_3d']
        gt_labels_3d = results['gt_labels_3d']
        
        # Convert to numpy arrays for processing
        if hasattr(points, 'tensor'):
            points_np = points.tensor.numpy()
        else:
            points_np = np.array(points)
            
        if hasattr(gt_bboxes_3d, 'tensor'):
            existing_boxes = gt_bboxes_3d.tensor.numpy()
        else:
            existing_boxes = gt_bboxes_3d
            
        if hasattr(gt_labels_3d, 'numpy'):
            existing_labels = gt_labels_3d.numpy()
        else:
            existing_labels = np.array(gt_labels_3d)
        
        # Lists to collect inserted objects
        inserted_points_list = []
        inserted_boxes_list = []
        inserted_labels_list = []
        
        successful_insertions = 0
        
        for exemplar in exemplars:
            # Get exemplar points from memory bank
            exemplar_points = self._get_exemplar_points(exemplar)
            if exemplar_points is None or len(exemplar_points) == 0:
                continue
                
            # Get exemplar bbox
            exemplar_bbox = self._get_exemplar_bbox(exemplar)
            if exemplar_bbox is None:
                continue
                
            # Find valid placement
            placed_points, placed_bbox = self._find_valid_placement(
                exemplar_points, exemplar_bbox, existing_boxes, points_np)
                
            if placed_points is not None and placed_bbox is not None:
                inserted_points_list.append(placed_points)
                inserted_boxes_list.append(placed_bbox)
                inserted_labels_list.append(exemplar['class_id'])
                successful_insertions += 1
        
        # Merge inserted objects with scene
        if successful_insertions > 0:
            results = self._merge_insertions_with_scene(
                results, inserted_points_list, inserted_boxes_list, inserted_labels_list)
            
            # Debug logging
            print(f"🎯 Inserted {successful_insertions}/{len(exemplars)} exemplars into scene")
        
        return results
    
    def _get_exemplar_points(self, exemplar):
        """Get point cloud for exemplar object from memory bank.
        
        Points are either cached in memory or re-extracted from ScanNet dataset.
        """
        # VALIDATION: Skip exemplars marked as failed during extraction
        if exemplar.get('extraction_failed', False):
            print(f"⚠️  [DEBUG] Skipping exemplar {exemplar.get('scene_id')}:{exemplar.get('object_idx')} - marked as extraction failed during population")
            return None
        
        # Check if exemplar has insufficient points from metadata
        if exemplar.get('point_count', 0) == 0:
            print(f"⚠️  [DEBUG] Skipping exemplar {exemplar.get('scene_id')}:{exemplar.get('object_idx')} - has 0 points in metadata")
            return None
            
        # Try cache first
        scene_id = exemplar['scene_id']
        object_idx = exemplar['object_idx'] 
        cache_key = (scene_id, object_idx)
        
        if hasattr(self.memory_bank, 'point_cloud_cache') and cache_key in self.memory_bank.point_cloud_cache:
            points = self.memory_bank.point_cloud_cache[cache_key]
            # VALIDATION: Ensure cached points are not empty
            if points is None or len(points) == 0:
                print(f"⚠️  Cached exemplar {scene_id}:{object_idx} has no points - will re-extract from ScanNet")
            else:
                print(f"✅ Retrieved cached exemplar {scene_id}:{object_idx} with {len(points)} points")
                return points
            
        # Use memory bank's get_exemplar_points which will re-extract from ScanNet if needed
        if hasattr(self.memory_bank, 'get_exemplar_points'):
            # Get dataset reference if available (will be set during dataset initialization)
            dataset_ref = getattr(self.memory_bank, 'dataset_ref', None)
            points = self.memory_bank.get_exemplar_points(exemplar, dataset_ref=dataset_ref)
            if points is not None and len(points) > 0:
                print(f"✅ Retrieved exemplar {scene_id}:{object_idx} with {len(points)} points")
                return points
            else:
                print(f"⚠️  [DEBUG] Exemplar {scene_id}:{object_idx} could not be loaded - extraction failed")
                return None
            
        print(f"⚠️  [DEBUG] Cannot retrieve points for exemplar {scene_id}:{object_idx} - memory bank missing get_exemplar_points")
        return None
    
    def _get_exemplar_bbox(self, exemplar):
        """Get bounding box for exemplar object."""
        bbox = exemplar.get('bbox', None)
        if bbox is None:
            return None
            
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox, dtype=np.float32)
            
        # Ensure 7D format (x, y, z, dx, dy, dz, yaw)
        if bbox.shape[-1] == 6:
            # Add zero yaw
            bbox = np.concatenate([bbox, [0.0]], axis=0)
        elif bbox.shape[-1] != 7:
            return None
            
        return bbox
    
    def _find_valid_placement(self, exemplar_points, exemplar_bbox, existing_boxes, scene_points):
        """Find valid placement for exemplar object with collision detection."""
        # Get scene bounds for placement constraints
        scene_min = scene_points.min(axis=0)[:3]  # x, y, z
        scene_max = scene_points.max(axis=0)[:3]
        
        # Center exemplar at origin for easier placement
        exemplar_center = exemplar_bbox[:3]
        centered_points = exemplar_points.copy()
        centered_points[:, :3] -= exemplar_center
        
        for attempt in range(self.max_placement_attempts):
            # Generate random placement position
            placement_x = random.uniform(
                scene_min[0] + exemplar_bbox[3]/2, 
                scene_max[0] - exemplar_bbox[3]/2)
            placement_y = random.uniform(
                scene_min[1] + exemplar_bbox[4]/2,
                scene_max[1] - exemplar_bbox[4]/2)
            placement_z = scene_min[2] + self.floor_offset + exemplar_bbox[5]/2
            
            # Add jitter
            placement_x += random.uniform(-self.placement_jitter, self.placement_jitter)
            placement_y += random.uniform(-self.placement_jitter, self.placement_jitter)
            
            placement_pos = np.array([placement_x, placement_y, placement_z])
            
            # Create placed bbox
            placed_bbox = exemplar_bbox.copy()
            placed_bbox[:3] = placement_pos
            
            # Check collision with existing objects
            if not self._check_collision(placed_bbox, existing_boxes):
                # Valid placement found
                placed_points = centered_points.copy()
                placed_points[:, :3] += placement_pos
                return placed_points, placed_bbox
                
        # No valid placement found
        return None, None
    
    def _check_collision(self, new_bbox, existing_boxes):
        """Check if new bbox collides with existing objects."""
        if len(existing_boxes) == 0:
            return False
            
        # Simple AABB collision check
        new_min = new_bbox[:3] - new_bbox[3:6] / 2
        new_max = new_bbox[:3] + new_bbox[3:6] / 2
        
        for existing_bbox in existing_boxes:
            if len(existing_bbox) < 6:
                continue
                
            existing_min = existing_bbox[:3] - existing_bbox[3:6] / 2
            existing_max = existing_bbox[:3] + existing_bbox[3:6] / 2
            
            # Check overlap in all 3 dimensions
            overlap_x = max(0, min(new_max[0], existing_max[0]) - max(new_min[0], existing_min[0]))
            overlap_y = max(0, min(new_max[1], existing_max[1]) - max(new_min[1], existing_min[1]))
            overlap_z = max(0, min(new_max[2], existing_max[2]) - max(new_min[2], existing_min[2]))
            
            # Calculate IoU approximation
            if overlap_x > 0 and overlap_y > 0 and overlap_z > 0:
                intersection = overlap_x * overlap_y * overlap_z
                new_volume = new_bbox[3] * new_bbox[4] * new_bbox[5]
                existing_volume = existing_bbox[3] * existing_bbox[4] * existing_bbox[5]
                union = new_volume + existing_volume - intersection
                
                if union > 0 and intersection / union > self.collision_threshold:
                    return True  # Collision detected
                    
        return False  # No collision
    
    def _merge_insertions_with_scene(self, results, inserted_points_list, inserted_boxes_list, inserted_labels_list):
        """Merge inserted exemplars with original scene data.
        
        IMPORTANT: Actual available attributes to avoid AttributeError:
        
        DepthPoints objects have:
        - tensor (torch.Tensor): Point data matrix
        - points_dim (int): Dimension of points
        - attribute_dims (dict): Extra dimension meanings  
        - rotation_axis (int): Default rotation axis (= 2 for DepthPoints)
        - coord (property): Coordinates of each point [:, :3]
        - height (property): Height of each point (if available)
        - color (property): Color of each point (if available)
        - shape (property): Shape of points tensor
        - device (property): Device of points tensor
        ❌ Does NOT have: coord_type
        
        DepthInstance3DBoxes objects have:
        - tensor (torch.Tensor): Box data matrix
        - box_dim (int): Box dimensions
        - with_yaw (bool): Whether has yaw rotation
        - volume (property): Volume of each box
        - dims (property): Size dimensions (N, 3)
        - yaw (property): Yaw of each box
        - height (property): Height of each box
        - top_height (property): Top height of each box
        - bottom_height (property): Bottom height of each box
        - center (property): Center of each box
        - bottom_center (property): Bottom center of each box
        - gravity_center (property): Gravity center of each box
        - corners (property): 8 corners of each box
        - bev (property): 2D BEV box with rotation
        - nearest_bev (property): 2D BEV box without rotation
        ❌ Does NOT have: box_mode
        
        Solution: Use type(object) directly instead of accessing non-existent attributes.
        """
        # Merge points
        original_points = results['points']
        if hasattr(original_points, 'tensor'):
            original_points_np = original_points.tensor.numpy()
        else:
            original_points_np = np.array(original_points)
            
        all_points = [original_points_np] + inserted_points_list
        merged_points_np = np.concatenate(all_points, axis=0)
        
        # Create new points object - use the same type as original points
        points_class = type(original_points)
        merged_points = points_class(
            merged_points_np,
            points_dim=merged_points_np.shape[-1],
            attribute_dims=getattr(original_points, 'attribute_dims', None)
        )
        
        # Merge bboxes
        original_bboxes = results['gt_bboxes_3d']
        if hasattr(original_bboxes, 'tensor'):
            original_boxes_np = original_bboxes.tensor.numpy()
        else:
            original_boxes_np = np.array(original_bboxes)
            
        inserted_boxes_np = np.array(inserted_boxes_list)
        merged_boxes_np = np.concatenate([original_boxes_np, inserted_boxes_np], axis=0)
        
        # Create new bbox object - use the same type as original bboxes
        box_type = type(original_bboxes)
        merged_bboxes = box_type(merged_boxes_np)
        
        # Merge labels  
        original_labels = results['gt_labels_3d']
        if hasattr(original_labels, 'numpy'):
            original_labels_np = original_labels.numpy()
        else:
            original_labels_np = np.array(original_labels)
            
        inserted_labels_np = np.array(inserted_labels_list)
        merged_labels_np = np.concatenate([original_labels_np, inserted_labels_np], axis=0)
        
        # CRITICAL FIX: Handle point masks to prevent IndexError in PointSample
        # When we add exemplar points, we must also add corresponding mask entries
        
        # Handle instance mask if present
        if 'pts_instance_mask' in results:
            original_instance_mask = results['pts_instance_mask']
            # Create instance masks for inserted points
            # Each inserted point gets a unique instance ID to distinguish from original instances
            total_inserted_points = sum(len(points) for points in inserted_points_list)
            if total_inserted_points > 0:
                # Use negative instance IDs to distinguish inserted exemplars from original instances
                inserted_instance_mask = np.arange(-1, -total_inserted_points - 1, -1, dtype=original_instance_mask.dtype)
                merged_instance_mask = np.concatenate([original_instance_mask, inserted_instance_mask], axis=0)
                results['pts_instance_mask'] = merged_instance_mask
                
        # Handle semantic mask if present  
        if 'pts_semantic_mask' in results:
            original_semantic_mask = results['pts_semantic_mask']
            # Create semantic masks for inserted points using their class labels
            inserted_semantic_masks = []
            for i, points in enumerate(inserted_points_list):
                # Each point in this exemplar gets the class label of the exemplar
                exemplar_class = inserted_labels_list[i]
                exemplar_semantic_mask = np.full(len(points), exemplar_class, dtype=original_semantic_mask.dtype)
                inserted_semantic_masks.append(exemplar_semantic_mask)
            
            if inserted_semantic_masks:
                inserted_semantic_mask = np.concatenate(inserted_semantic_masks, axis=0)
                merged_semantic_mask = np.concatenate([original_semantic_mask, inserted_semantic_mask], axis=0)
                results['pts_semantic_mask'] = merged_semantic_mask
        
        # Update results
        results['points'] = merged_points
        results['gt_bboxes_3d'] = merged_bboxes
        results['gt_labels_3d'] = merged_labels_np
        
        return results
    
    def __repr__(self):
        """String representation."""
        return (f'{self.__class__.__name__}('
                f'max_exemplars_per_scene={self.max_exemplars_per_scene}, '
                f'insertion_probability={self.insertion_probability}, '
                f'collision_threshold={self.collision_threshold})')