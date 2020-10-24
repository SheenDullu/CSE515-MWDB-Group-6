import Task2
import Task3
import Utilities
import task1
from Task0 import task0a, task0b


def main():
    while True:
        print("########## Phase 1 ##########")
        print("Task 0a: Perform Normalization and Quantization")
        print("Task 0b: Build Gesture Vectors")
        print("Task 1: Build the PCA, SVD, NMF, LDA word scores")
        print("Task 2: 10 most similar gestures")
        print("Task 3: Latent Gesture Discovery Tasks")
        print("Task 4: Latent Gesture Clustering and Analysis Tasks")
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
            task0b(Utilities.read_directory())
            print("########## Completed Task 2 ##########")
        elif task == '1':
            task1.main()
        elif task == '2':
            Task2.main()
        elif task == '3':
            Task3.main()
        # elif task == '4':

        elif task == '0':
            exit()


if __name__ == '__main__':
    main()
