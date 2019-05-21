from keras.layers import Input, Embedding, Dot, Reshape, Add
from keras.models import Model
from keras.optimizers import Adam
import keras.backend as k


def glove_model(vocab_size=10, vector_dim=3):
    """
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

    prediction = Add()([dot_product, target_bias, context_bias])

    model = Model(inputs=[input_target, input_context], outputs=prediction)
    model.compile(loss=custom_loss, optimizer=Adam())

    return model


def custom_loss(y_true, y_pred):
    """
    This is GloVe's loss function, view section 3.1 on the original paper for details.
    :param y_true: The actual values, y_true = X_ij
    :param y_pred: The predicted occurrences from the model ( w_i^T*w_j )
    :return: The loss associated with this batch
    """
    x_max = 100
    alpha = 3.0 / 4.0
    fxij = k.pow(k.clip(y_true / x_max, 0.0, 1.0), alpha)

    return k.sum(fxij * k.square(y_pred - k.log(y_true)), axis=-1)
