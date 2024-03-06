import shutil
import os

def merge_subfolders(source_folder, destination_folder):
    try:
        # Check if the destination folder exists, if not, create it
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Get a list of subfolders in the source folder
        subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

        # Loop through each subfolder and merge its contents into the destination folder
        for subfolder in subfolders:
            for root, _, files in os.walk(subfolder):
                for file in files:
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(destination_folder, file)
                    shutil.move(source_file, destination_file)  # Use shutil.copy() if you want to copy instead of move

        print("Subfolders merged successfully!")
    except shutil.Error as e:
        print(f"Error occurred: {e}")
    except OSError as e:
        print(f"Error: {e}")


def get_img_array(img):

    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

