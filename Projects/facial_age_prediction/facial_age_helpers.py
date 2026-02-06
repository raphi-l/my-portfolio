
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0, EfficientNetB1
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau

import datetime

IMAGE_SHAPE = (224,224)
BATCH_SIZE = 32


def prepare_face_data(df: pd.DataFrame = None,
                      df_path: str = None,
                      test_size: int = 0.2,
                      min_age_count: int = 5):
  """
  Returns two DataFrames split into training and testing DataFrames
  Takes either DataFrame or path to csv file. Default is 20% test size
  """
  if df:
    pass
  else:
    df = pd.read_csv(df_path)

  age_counts=df['real_age'].value_counts()
  repeated_ages = age_counts[age_counts > min_age_count].index
  df_filtered = df[df['real_age'].isin(repeated_ages)]
  
  train_df, test_df = train_test_split(df_filtered,
                                     shuffle=True,
                                     random_state=42, test_size=0.2,
                                     stratify=df_filtered['real_age'])
  return train_df, test_df


def load_train(base_path, df, range_255= False):

    """
    It loads the train part of dataset from path
    """

    if range_255 == False:
        train_gen = ImageDataGenerator(rescale=1/255.)
    else:
        train_gen = ImageDataGenerator()

    print(f"[INFO] Loading Testing Images using Color-Channel Range {'[0,1]' if not range_255 else '[0,255]'}")

    df['full_path'] = df['file_name'].apply(lambda x: os.path.join(base_path, x))

    train_gen_flow = train_gen.flow_from_dataframe(
        dataframe=df,
        x_col='full_path',
        y_col='real_age',
        target_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='raw',
        shuffle=True
    )

    return train_gen_flow


def load_test(base_path, df, range_255=False):

    """
    It loads the validation/test part of dataset from path
    """

    if range_255 == False:
        test_gen = ImageDataGenerator(rescale=1/255.)
    else:
        test_gen = ImageDataGenerator()

    print(f"[INFO] Loading Testing Images using Color-Channel Range {'[0,1]' if not range_255 else '[0,255]'}")

    df['full_path'] = df['file_name'].apply(lambda x: os.path.join(base_path, x))

    test_gen_flow = test_gen.flow_from_dataframe(
        dataframe=df,
        x_col='full_path',
        y_col='real_age',
        target_size=IMAGE_SHAPE,
        batch_size=BATCH_SIZE,
        class_mode='raw',
        shuffle=True
    )

    return test_gen_flow


def create_model(base_model_name:str,
                 input_shape= IMAGE_SHAPE+(3,),
                 num_classes=1):

    """
    It defines the model
    """

    model_dict = {
        'ResNet50' : ResNet50,
        'EfficientNetB0': EfficientNetB0,
        'EfficientNetB1': EfficientNetB1
    }

    if base_model_name not in model_dict:
        raise ValueError(
          f"Unsupported base model: {base_model_name}."
          f'Available options: {list(model_dict).keys()}'
      )

    base_model = model_dict[base_model_name](include_top=False,
                                             weights='imagenet',
                                             input_shape=input_shape)

    base_model.trainable=False

    transfer_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation='relu', name='output_layer')])

    return transfer_model


def train_model(model,
                model_name:str,
                train_data: ImageDataGenerator,
                test_data: ImageDataGenerator,
                batch_size: int = 32,
                epochs: int = 20,
                steps_per_epoch: int = None,
                validation_steps: int = None,
                log_dir:str = None,
                patience_lr: int = 3,
                min_delta_lr: float = 0.05,
                min_lr: float = 1e-5):

    """
    Trains the model given the parameters
    """

    model.compile(loss='mse',
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['mae'])

    reduce_lr = ReduceLROnPlateau(
        monitor='val_mae',
        factor=0.5,
        patience=patience_lr,
        min_delta=min_delta_lr,
        min_lr=min_lr,
        verbose=1)

    print(f'[INFO] Training Model...')

    model_history = model.fit(train_data,
                             epochs=epochs,
                             steps_per_epoch=steps_per_epoch,
                             validation_data=test_data,
                             validation_steps=validation_steps,
                             callbacks=[reduce_lr,
                                        create_tensorboard_callback(dir_name='tensorflow_hub',
                                                                    experiment_name=model_name,
                                                                    log_dir=log_dir)])

    return model


def create_tensorboard_callback(dir_name, experiment_name, log_dir=None):
  if log_dir is None:
    log_dir = dir_name + '/' + experiment_name + '/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f"Saving TensorBoard log files to: {log_dir}")

  return tensorboard_callback


