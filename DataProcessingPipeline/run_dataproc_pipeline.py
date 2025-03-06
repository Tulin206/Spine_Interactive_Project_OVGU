import DataExtraction as de
import GaitProcessing as gp
import FeatureExtraction_old as fe
from visualize.visualization import plot_signals, plot_signal_peaks

# DataProcessingPipeline
# Read data -> Detect Gait Events -> Segmentation -> Filtering ->
# Transforming to Global Coordinate System

# enode_data_dir = "/data/project/spine/enode_data/"
enode_data_dir = "/home/isratjahantulin/Downloads/all_data/all_data/Enode/"  ## ISRAT's Location
# sp_data_dir = "/data/project/spine/smartphone_data/"
sp_data_dir = "/home/isratjahantulin/Downloads/all_data/all_data/Smartphone/"   ## ISRAT's Location
data_dir = {"enode": enode_data_dir, "phone": sp_data_dir}
fs = 100
test = "gait"     # Added by Israt

def run():
    # Data Extraction & Transforming
    data = de.DataExtraction(data_dir).run()

    # Detect Gait Events & Segmentation & Filter
    data_s = gp.GaitProcessing(data).run(test)
    plot_signal_peaks(data_s["signal"], data_s["h"], title="Shank Peaks")
    plot_signal_peaks(data_s["signal"], data_s["h"], title="Shank Peaks")


    # Feature Extraction
    #f_extract = fe.FeatureExtraction(cog)
    #combined_features = f_extract.run()

    #return combined_features, y


if __name__ == "__main__":
    run()
