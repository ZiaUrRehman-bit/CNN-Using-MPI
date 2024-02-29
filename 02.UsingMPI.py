import numpy as np
from mpi4py import MPI
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')
    ])
    return model
# Define the local CNN model
def create_local_model():
    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, x_train, y_train):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2, verbose=0)

def main():
    # Load and preprocess the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1) / 255.0  # Normalize pixel values
    x_test = np.expand_dims(x_test, axis=-1) / 255.0
    y_train = to_categorical(y_train, num_classes=10)  # One-hot encode labels
    y_test = to_categorical(y_test, num_classes=10)

    # Distribute data among MPI processes (optional)
    chunk_size = len(x_train) // size
    start_idx = rank * chunk_size
    end_idx = (rank + 1) * chunk_size if rank < size - 1 else len(x_train)
    x_train_chunk = x_train[start_idx:end_idx]
    y_train_chunk = y_train[start_idx:end_idx]
    print(f"Rank: {rank}, Training data Size: X--> {x_train_chunk.shape}, y--> {y_train_chunk.shape}" )
    # Build the model
    model = build_model()

    startTime = time.time()
    # Train the model in parallel
    train_model(model, x_train_chunk, y_train_chunk)
    endTime = time.time()
    totalTime = endTime - startTime
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"Rank: {rank}, Test Loss: {test_loss}")
    print(f"Rank: {rank}, Test Accuracy:{test_accuracy}")
    print(f"Total Time Taken by process {rank} is {totalTime}")
    
    gatherdTime = comm.gather(totalTime, root=4)
    getherdTestAccuracy = comm.gather(test_accuracy, root=4)

    # weights = model.get_weights()

    # gatherWeights = comm.gather(weights, root=4)

    if rank == 4:
        print(f"Total time: {gatherdTime}")
        print(f"Average Time: {sum(gatherdTime)/size}")
        print(f"Test Accuracy: {getherdTestAccuracy}")
        print(f"Average Test Accuracy: {sum(getherdTestAccuracy)/size}")
        


if __name__ == "__main__":
    main()
