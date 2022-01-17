from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


def get_callbacks():
    ea_stop, checkpoint = EarlyStopping(monitor='val_loss', patience=5, mode='min'), ModelCheckpoint(
        f'models/new_models_with_top_3k',
        filename='{epoch}-{val_loss:.3f}',
        monitor='val_loss',
        save_top_k=3)
    callbacks = ea_stop
    return callbacks, checkpoint