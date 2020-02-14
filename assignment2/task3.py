import numpy as np
import utils
import matplotlib.pyplot as plt
import typing
import pickle
from task2a import cross_entropy_loss, SoftmaxModel, one_hot_encode, pre_process_images
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray,
                       model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    targets_indices = np.argmax(targets, axis = 1)
    outputs_indices = np.argmax(model.forward(X), axis = 1)
    result = np.equal(targets_indices, outputs_indices)
    result.size
    accuracy = (result.sum()) / result.size
    return accuracy


def unison_shuffled_copies(X, targets):
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
    Returns:
        unison shuffled copies of X and targets 
    """
    assert len(X) == len(targets)
    p = np.random.permutation(len(X))
    return X[p], targets[p]

def train(
        model: SoftmaxModel,
        datasets: typing.List[np.ndarray],
        num_epochs: int,
        learning_rate: float,
        batch_size: int,
        # Task 3 hyperparameters,
        use_shuffle: bool,
        use_momentum: bool,
        momentum_gamma: float):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = datasets

    # Utility variables
    num_batches_per_epoch = X_train.shape[0] // batch_size
    num_steps_per_val = num_batches_per_epoch // 5
    # Tracking variables to track loss / accuracy
    train_loss = {}
    val_loss = {}
    train_accuracy = {}
    val_accuracy = {}
    
    if use_momentum:
        learning_rate = 0.02

    global_step = 0
    for epoch in range(num_epochs):
        for step in range(num_batches_per_epoch):
            start = step * batch_size
            end = start + batch_size
            X_batch, Y_batch = X_train[start:end], Y_train[start:end]

            # Track train / validation loss / accuracy
            # every time we progress 20% through the dataset
            prev_grads = model.grads

            outputs = model.forward(X_batch)
            model.backward(X_batch, outputs, Y_batch)
            for i in range(len(model.ws)):
                if use_momentum:
                    model.ws[i] = model.ws[i] - learning_rate  * (model.grads[i] + momentum_gamma * prev_grads[i])
                else:
                    model.ws[i] = model.ws[i] - learning_rate * model.grads[i]

            if (global_step % num_steps_per_val) == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                _train_loss = cross_entropy_loss(Y_train, model.forward(X_train))
                train_loss[global_step] = _train_loss

                train_accuracy[global_step] = calculate_accuracy(
                    X_train, Y_train, model)
                val_accuracy[global_step] = calculate_accuracy(
                    X_val, Y_val, model)

            global_step += 1
        # shuffle training examples after each epoch
        if use_shuffle:
            X_train, Y_train = unison_shuffled_copies(X_train, Y_train)
    return model, train_loss, val_loss, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Load dataset
    validation_percentage = 0.2
    X_train, Y_train, X_val, Y_val, X_test, Y_test = utils.load_full_mnist(
        validation_percentage)

    #preprocessing of targets and images
    x_train_mean = np.mean(X_train)
    x_train_std = np.std(X_train)
    X_train = pre_process_images(X_train, x_train_mean, x_train_std)
    X_test = pre_process_images(X_test, x_train_mean, x_train_std)
    X_val = pre_process_images(X_val, x_train_mean, x_train_std)
    Y_train = one_hot_encode(Y_train, 10)
    Y_test = one_hot_encode(Y_test, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # Hyperparameters
    num_epochs = 50
    learning_rate = .1
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter

    # Settings for task 3. Keep all to false for task 2.
    use_shuffle = True
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    model, train_loss, val_loss, train_accuracy, val_accuracy = train(
        model,
        [X_train, Y_train, X_val, Y_val, X_test, Y_test],
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        use_shuffle=use_shuffle,
        use_momentum=use_momentum,
        momentum_gamma=momentum_gamma)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Test Cross Entropy Loss:",
          cross_entropy_loss(Y_test, model.forward(X_test)))

    print("Final Train accuracy:",
          calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:",
          calculate_accuracy(X_val, Y_val, model))
    print("Final Test accuracy:",
          calculate_accuracy(X_test, Y_test, model))

    # Use pickle to save the losses and accuracies
    savename = 'momentum_3d'
    with open('data/train_loss_' + savename + '.pickle', 'wb') as file:
        pickle.dump(train_loss, file)
    with open('data/val_loss_' + savename + '.pickle', 'wb') as file:
        pickle.dump(val_loss, file)
    with open('data/train_accuracy_' + savename + '.pickle', 'wb') as file:
        pickle.dump(train_accuracy, file)
    with open('data/val_accuracy_' + savename + '.pickle', 'wb') as file:
        pickle.dump(val_accuracy, file)

    # Plot loss
    plt.figure(figsize=(20, 8))

    plot_base = False
    plot_shuffle = False 
    plot_sigma = False
    plot_weights = True
    plot_momentum = True

    plt.subplot(1, 2, 1)
    if plot_base:
        train_loss_base = []
        val_loss_base = []
        with open('data/train_loss_base_2c.pickle', 'rb') as file:
            train_loss_base = pickle.load(file)
        with open('data/val_loss_base_2c.pickle', 'rb') as file:
            val_loss_base = pickle.load(file)
        utils.plot_loss(train_loss_base, "Training Loss Base", fmt = '-g')
        utils.plot_loss(val_loss_base, "Validation Loss Base", fmt = '--g')

    if plot_shuffle:
        train_loss_shuffle = []
        val_loss_shuffle = []
        with open('data/train_loss_shuffle_3a.pickle', 'rb') as file:
            train_loss_shuffle = pickle.load(file)
        with open('data/val_loss_shuffle_3a.pickle', 'rb') as file:
            val_loss_shuffle = pickle.load(file)
        utils.plot_loss(train_loss_shuffle, "Training Loss Shuffle", fmt = '-b')
        utils.plot_loss(val_loss_shuffle, "Validation Loss Shuffle", fmt = '--b')

    if plot_sigma:
        train_loss_sigma = []
        val_loss_sigma = []
        with open('data/train_loss_sigma_3b.pickle', 'rb') as file:
            train_loss_sigma = pickle.load(file)
        with open('data/val_loss_sigma_3b.pickle', 'rb') as file:
            val_loss_sigma = pickle.load(file)
        utils.plot_loss(train_loss_sigma, "Training Loss Improved Sigma", fmt = '-r')
        utils.plot_loss(val_loss_sigma, "Validation Loss Improved Sigma", fmt = '--r')
        
    if plot_weights:
        train_loss_weights = []
        val_loss_weights = []
        with open('data/train_loss_weights_3c.pickle', 'rb') as file:
            train_loss_weights = pickle.load(file)
        with open('data/val_loss_weights_3c.pickle', 'rb') as file:
            val_loss_weights = pickle.load(file)
        utils.plot_loss(train_loss_weights, "Training Loss Weights Initialization", fmt = '-k')
        utils.plot_loss(val_loss_weights, "Validation Loss Weights Initialization", fmt = '--k')

    if plot_momentum:
        train_loss_momentum = []
        val_loss_momentum = []
        with open('data/train_loss_momentum_3d.pickle', 'rb') as file:
            train_loss_momentum = pickle.load(file)
        with open('data/val_loss_momentum_3d.pickle', 'rb') as file:
            val_loss_momentum = pickle.load(file)
        utils.plot_loss(train_loss_momentum, "Training Loss Momentum", fmt = '-m')
        utils.plot_loss(val_loss_momentum, "Validation Loss Momentum", fmt = '--m')

    plt.ylim([0.1, .5])
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
        
    # PLotting Accuracy   

    plt.subplot(1, 2, 2)

    if plot_base: 
        train_accuracy_base = []
        val_accuracy_base = []
        with open('data/train_accuracy_base_2c.pickle', 'rb') as file:
            train_accuracy_base = pickle.load(file)
        with open('data/val_accuracy_base_2c.pickle', 'rb') as file:
            val_accuracy_base = pickle.load(file)
        utils.plot_loss(train_accuracy_base, "Training Accuracy Base", fmt = '-g')
        utils.plot_loss(val_accuracy_base, "Validation Accuracy Base", fmt = '--g')
        
    if plot_shuffle: 
        train_accuracy_shuffle = []
        val_accuracy_shuffle = []
        with open('data/train_accuracy_shuffle_3a.pickle', 'rb') as file:
            train_accuracy_shuffle = pickle.load(file)
        with open('data/val_accuracy_shuffle_3a.pickle', 'rb') as file:
            val_accuracy_shuffle = pickle.load(file)
        utils.plot_loss(train_accuracy_shuffle, "Training Accuracy Shuffle", fmt = '-b')
        utils.plot_loss(val_accuracy_shuffle, "Validation Accuracy Shuffle", fmt = '--b')

    if plot_sigma: 
        train_accuracy_sigma = []
        val_accuracy_sigma = []
        with open('data/train_accuracy_sigma_3b.pickle', 'rb') as file:
            train_accuracy_sigma = pickle.load(file)
        with open('data/val_accuracy_sigma_3b.pickle', 'rb') as file:
            val_accuracy_sigma = pickle.load(file)
        utils.plot_loss(train_accuracy_sigma, "Training Accuracy Improved Sigma", fmt = '-r')
        utils.plot_loss(val_accuracy_sigma, "Validation Accuracy Improved Sigma", fmt = '--r')

    if plot_weights: 
        train_accuracy_sigma = []
        val_accuracy_sigma = []
        with open('data/train_accuracy_weights_3c.pickle', 'rb') as file:
            train_accuracy_sigma = pickle.load(file)
        with open('data/val_accuracy_weights_3c.pickle', 'rb') as file:
            val_accuracy_sigma = pickle.load(file)
        utils.plot_loss(train_accuracy_sigma, "Training Accuracy Weights Initialization", fmt = '-k')
        utils.plot_loss(val_accuracy_sigma, "Validation Accuracy Weights Initialization", fmt = '--k')

    if plot_momentum:
        train_loss_momentum = []
        val_loss_momentum = []
        with open('data/train_accuracy_momentum_3d.pickle', 'rb') as file:
            train_accuracy_momentum = pickle.load(file)
        with open('data/val_accuracy_momentum_3d.pickle', 'rb') as file:
            val_accuracy_momentum = pickle.load(file)
        utils.plot_loss(train_accuracy_momentum, "Training Accuracy Momentum", fmt = '-m')
        utils.plot_loss(val_accuracy_momentum, "Validation Accuracy Momentum", fmt = '--m')

    plt.ylim([0.9, 1.0])
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    # utils.plot_loss(train_accuracy_prev, "Training Accuracy Shuffled", fmt = '-g')
    # utils.plot_loss(val_accuracy_prev, "Validation Accuracy Shuffled", fmt = '--g')
    plt.savefig("softmax_train_graph.png")
    plt.show()
