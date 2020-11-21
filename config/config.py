TRAIN_IMAGES_PATH = "../datasets/tiny-imagenet-200/train"
VALID_IMAGES_PATH = "../datasets/tiny-imagenet-200/val/images"

# Maps valid image names to WordNetID
VALID_MAPPING_PATH = "../datasets/tiny-imagenet-200/val/val_annotations.txt"

# List of all Wordnet IDs
WORDNET_ID_PATH = "../datasets/tiny-imagenet-200/wnids.txt"
# Maps Wordnet ID to classnames
WORDNET_LABEL_PATH = "../datasets/tiny-imagenet-200/words.txt"

NUM_CLASSES = 200
NUM_TEST_IMAGES = 50 * NUM_CLASSES

TRAIN_PATH = "../datasets/tiny-imagenet-200/hdf5/train.hdf5"
VALID_PATH = "../datasets/tiny-imagenet-200/hdf5/valid.hdf5"
TEST_PATH = "../datasets/tiny-imagenet-200/hdf5/test.hdf5"

OUTPUT_PATH = "output"
MODEL_PATH = "output/model"
MEAN_PATH = "output/mean.json"
FIG_PATH = "output/training.png"
JSON_PATH = "output/training.json"
