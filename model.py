from tensorflow.python.keras.layers import Input, Embedding, Dot, Reshape
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.optimizers import Adam
import tensorflow.python.keras.backend as K


def glove_model(vocab_size=10, vector_dim=3):
    """
    A Keras implementation of the GloVe architecture
    :param vocab_size: The number of distinct words
    :param vector_dim: The vector dimension of each word
    :return:
    """
    input_target = Input((1,))
    input_context = Input((1,))

    target_embedding = Embedding(vocab_size, vector_dim, input_length=1)(input_target)
    target_bias = Embedding(vocab_size, 1, input_length=1)(input_target)

    context_embedding = Embedding(vocab_size, vector_dim, input_length=1)(input_context)
    context_bias = Embedding(vocab_size, 1, input_length=1)(input_context)

    dot_product = Dot(axes=-1)([target_embedding, context_embedding])
    dot_product = Reshape((1, ))(dot_product)
    target_bias = Reshape((1,))(target_bias)
    contex_bias = Reshape((1,))(context_bias)

    model = Model(inputs=[input_target, input_context], outputs=[dot_product, target_bias, contex_bias])
    model.compile(loss=custom_loss, optimizer=Adam())

    return model


def custom_loss(y_true, y_pred):
    """
    This is GloVe's loss function
    :param y_true: The actual values, y_true = X_ij
    :param y_pred: The predicted occurrences from the model ( w_i^T*w_j )
    :return: The loss associated with this batch
    """
    x_max = 100
    alpha = 3.0 / 4.0
    fxij = K.pow(K.clip(y_true / x_max, 0.0, 1.0), alpha)

    return K.sum(fxij * K.square(y_pred - K.log(y_true)), axis=-1)
