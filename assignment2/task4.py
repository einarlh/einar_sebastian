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

            if (epoch % 5) == 0 and step == 0:
                print("Training at epoch: " + str(epoch))

            if (global_step % num_steps_per_val) == 0:
                _val_loss = cross_entropy_loss(Y_val, model.forward(X_val))
                val_loss[global_step] = _val_loss

                #change to train
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


def plot_loss(basename, displayname, fmt_string_train, fmt_string_val):
    train_loss_loaded = []
    val_loss_loaded = []
    train_path = 'data/train_loss_' + basename + '.pickle'
    val_path = 'data/val_loss_' + basename + '.pickle'
    with open(train_path, 'rb') as file:
        train_loss_loaded = pickle.load(file)
    with open(val_path, 'rb') as file:
        val_loss_loaded = pickle.load(file)
    utils.plot_loss(train_loss_loaded, "Training Loss " + displayname, fmt = fmt_string_train)
    utils.plot_loss(val_loss_loaded, "Validation Loss " + displayname, fmt = fmt_string_val)


def plot_accuracy(basename, displayname, fmt_string_train, fmt_string_val):
    train_accuracy_loaded = []
    val_accuracy_loaded = []
    train_path = 'data/train_accuracy_' + basename + '.pickle'
    val_path = 'data/val_accuracy_' + basename + '.pickle'

    with open(train_path, 'rb') as file:
        train_accuracy_loaded = pickle.load(file)

    with open(val_path, 'rb') as file:
        val_accuracy_loaded = pickle.load(file)
    utils.plot_loss(train_accuracy_loaded, "Training Accuracy " + displayname, fmt = fmt_string_train)
    utils.plot_loss(val_accuracy_loaded, "Validation Accuracy " + displayname, fmt = fmt_string_val)

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
    neurons_per_layer =  [60, 60, 10]
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

#    Use pickle to save the losses and accuracies
    savename = 'double_60_60'
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

    plot_base = True
    plot_shuffle = False 
    plot_sigma = False
    plot_weights = False
    plot_momentum = False
    plot_double_hidden_60 = False
    plot_hidden_16 = False
    plot_hidden_128 = False

    plt.subplot(1, 2, 1)

    if plot_base: 
        plot_loss('base_2c', '', '-g', '--b')

    if plot_shuffle: 
        plot_loss('shuffle_3a', 'Shuffle', '-b', '--b')

    if plot_sigma: 
        plot_loss('sigma_3b', 'Improved Sigma', '-r', '--r')

    if plot_weights: 
        plot_loss('weights_3c', 'Weights Initialization', '-k', '--k')

    if plot_momentum:
        plot_loss('momentum_3d', 'Momentum', '-m', '--m')

    if plot_hidden_16: 
        plot_loss('hidden_16', 'Single Hidden 16', '-y', '--y')

    if plot_hidden_128: 
        plot_loss('hidden_128', 'Single Hidden 128', '-c', '--c')

    if plot_double_hidden_60:
        plot_loss('double_60_60', 'Double Hidden 60', '-y', '--y')

    plt.ylim([0, .5])
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
        
    # PLotting Accuracy   

    plt.subplot(1, 2, 2)

    if plot_base: 
        plot_accuracy('base_2c', 'Base', '-g', '--b')

    if plot_shuffle: 
        plot_accuracy('shuffle_3a', 'Shuffle', '-b', '--b')

    if plot_sigma: 
        plot_accuracy('sigma_3b', 'Improved Sigma', '-r', '--r')

    if plot_weights: 
        plot_accuracy('weights_3c', 'Weights Initialization', '-k', '--k')

    if plot_momentum:
        plot_accuracy('momentum_3d', 'Momentum', '-m', '--m')

    if plot_double_hidden_60:
        plot_accuracy('double_60_60', 'Double Hidden 60', '-y', '--y')

    if plot_hidden_16: 
        plot_accuracy('hidden_16', 'Single Hidden 16', '-y', '--y')

    if plot_hidden_128: 
        plot_accuracy('hidden_128', 'Single Hidden 128', '-c', '--c')


    plt.ylim([0.9, 1.0])
    plt.legend()
    plt.xlabel("Number of gradient steps")
    plt.ylabel("Accuracy")
    # utils.plot_loss(train_accuracy_prev, "Training Accuracy Shuffled", fmt = '-g')
    # utils.plot_loss(val_accuracy_prev, "Validation Accuracy Shuffled", fmt = '--g')
    plt.savefig("softmax_train_graph.png")
    plt.show()
