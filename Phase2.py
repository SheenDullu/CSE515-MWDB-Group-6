import Utilities
import task1
from Task0 import task0a, task0b


def main():
    while True:
        print("########## Phase 1 ##########")
        print("Task 0a: Perform Normalization and Quantization")
        print("Task 0b: Build Gesture Vectors")
        print("Task 1: Build the word score")
        task = input("What Task do you want to perform: (enter 0 to exit)\n")
        if task == '0a':
            folder_directory = input("Input Directory path of the gesture folders:\n")
            window_length = int(input("Enter window length(w): \n"))
            shift_length = int(input("Enter shift length(s): \n"))
            resolution = int(input("Enter resolution (r): \n"))
            Utilities.store_directory(folder_directory)
            task0a(folder_directory, window_length, shift_length, resolution)
            print("########## Completed Task 1 ##########")
        elif task == '0b':
            # directory = input("Input Directory path of the gesture folders:\n")
            task0b(Utilities.read_directory())
            print("########## Completed Task 2 ##########")
        elif task == '1':
            task1.main()
        elif task == '0':
            exit()


if __name__ == '__main__':
    main()
