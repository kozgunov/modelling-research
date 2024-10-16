import numpy as np

class RoundCounter:
    def __init__(self, num_buoys):
        self.num_buoys = num_buoys
        self.ship_positions = {}
        self.completed_rounds = {}
        self.buoy_positions = self._initialize_buoy_positions()

    def _initialize_buoy_positions(self):
        # the method should return a list of buoy positions (3-6 buoys positions)
        angles = np.linspace(0, 2*np.pi, self.num_buoys, endpoint=False)
        return [(np.cos(angle), np.sin(angle)) for angle in angles]

    def update(self, tracked_objects, camera_id):
        for obj in tracked_objects:
            if obj['class'] == 6:
                ship_id = obj['id']
                position = self._get_position(obj['bbox'], camera_id)
                
                if ship_id not in self.ship_positions:
                    self.ship_positions[ship_id] = [position]
                    self.completed_rounds[ship_id] = 0
                else:
                    self.ship_positions[ship_id].append(position)
                    
                    if self._check_round_completion(self.ship_positions[ship_id]):
                        self.completed_rounds[ship_id] += 1
                        self.ship_positions[ship_id] = [position]  # reset the positions of models after completing a round

    def _get_position(self, bbox, camera_id):
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return (center_x, center_y) # the position of the model in the middle of the box

    def _check_round_completion(self, positions):
        if len(positions) < self.num_buoys:
            return False
        
        # check if the ship has passed near all 6 buoys in correct order (from outside)
        buoy_index = 0
        for position in positions:
            if self._is_near_buoy(position, self.buoy_positions[buoy_index]):
                buoy_index += 1
                if buoy_index == self.num_buoys:
                    return True
        return False

    def _is_near_buoy(self, position, buoy_position, threshold=0.1):
        # check if a position is near a buoy
        distance = np.sqrt((position[0] - buoy_position[0])**2 + (position[1] - buoy_position[1])**2)
        return distance < threshold

    def get_results(self):
        return self.completed_rounds
