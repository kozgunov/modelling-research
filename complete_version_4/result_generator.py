import cv2
import numpy as np
import matplotlib.pyplot as plt


class ResultGenerator:
    def generate(self, results, video_paths):
        self._generate_summary_video(results, video_paths) # video input
        self._generate_text_output(results) # stats
        self._generate_visualization(results) # plot

    def _generate_summary_video(self, results, video_paths):
        # ------------------ video with outlines of all the objects ---------------------------------------------------
        pass

    def _generate_text_output(self, results):
        with open('race_results.txt', 'w') as f:
            for ship_id, rounds in results.items():
                f.write(f"Ship {ship_id}: {rounds} rounds completed\n")
        # -----------------------------------------------------add more stats -----------------------------------------

    def _generate_visualization(self, results):
        ship_ids = list(results.keys()) # counting
        rounds = list(results.values()) # counting

        plt.figure(figsize=(12, 6))
        plt.bar(ship_ids, rounds)
        plt.title('Race Results') # photos with outline
        plt.xlabel('Ship ID')
        plt.ylabel('Completed Rounds')
        plt.savefig('race_results_visualization.png')
        plt.close()


    def generate_live_leaderboard(self, results):
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True) # showing leaderboard during the race goes on 
        return sorted_results
