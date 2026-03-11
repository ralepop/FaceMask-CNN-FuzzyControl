import os
import matplotlib.pyplot as plt
from data_loader import extract_data, get_class_distribution, get_data_generators
from model_builder import build_cnn, compile_model

ZIP_PATH = 'dataset.zip'
DATASET_DIR = 'dataset/'
MODEL_SAVE_PATH = 'models/mask_detector_v1.keras'
EPOCHS = 15
BATCH_SIZE = 32

def main():
    extract_data(ZIP_PATH, DATASET_DIR)

    # histogram
    get_class_distribution(DATASET_DIR)

    # generatori
    train_gen, val_gen = get_data_generators(DATASET_DIR, batch_size = BATCH_SIZE)

    # kreiranje modela
    model = build_cnn(input_shape = (128, 128, 3))
    model = compile_model(model)

    # ispis
    model.summary()

    # trening
    print("Pocetak treninga...")
    if not os.path.exists('models'):
        os.makedirs('models')
    model.save(MODEL_SAVE_PATH)
    print(f"Model sacuvan na: {MODEL_SAVE_PATH}")

    history = model.fit(
        train_gen,
        validation_data = val_gen,
        epochs = EPOCHS
    )

    # grafik performansi
    plt.figure(figsize = (12, 4))

    # tacnost
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label = 'Trening Tacnost')
    plt.plot(history.history['val_accuracy'], label = 'Validaciona Tacnost')
    plt.title('Tacnost tokom epoha')
    plt.legend()

    # gubitak (loss)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label = 'Trening Gubitak')
    plt.plot(history.history['val_loss'], label = 'Validacioni Gubitak')
    plt.title('Gubitak tokom epoha')
    plt.legend()

    plt.savefig('reports/performance_graphs.png')
    plt.show()

if __name__ == "_main__":
    main()