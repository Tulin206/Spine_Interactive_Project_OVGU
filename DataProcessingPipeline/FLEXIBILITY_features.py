from DataProcessingPipeline.FeatureExtraction import FeatureExtraction
"""
we don't need inherit FeatureExtraction class rather we can create an instance of FeatureExtraction class to access it's attributes (composition principle-> "has-a" relationship)
class TUG_features(FeatureExtraction):
    def __init__(self, acc, gyr, events, fs, unproc_acc):
        # Initialize the parent FeatureExtraction class
        super().__init__(acc, gyr, events, fs, unproc_acc)
"""
class FLEXIBILITY_features:
    def __init__(self, feature_extractor: FeatureExtraction):
        self.fe = feature_extractor  # Store the FeatureExtraction instance

    def get_flexibility_features(self):
        ...
        flex = 0
        flex_list = "max_gyr"
        return flex, flex_list