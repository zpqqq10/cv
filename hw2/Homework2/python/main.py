from feature import Processor
from dataset import load_data, makedirs

if __name__ == "__main__":
    processor = Processor(load_data())
    makedirs('../result')
    processor.panorama_stitch('../result')
    # processor.visualize_keypoints('../result')
    # processor.visualize_matches('../result')
    processor.visualize_comparison('../result')
    