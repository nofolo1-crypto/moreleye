#!/usr/bin/env python3
"""
MorelEye — Model Training Pipeline
===================================
Trains custom classification heads on top of MobileNet v3 Small
for each species pack. Exports to TensorFlow.js format for
on-device inference.

Usage:
  pip install tensorflow tensorflow-hub tensorflowjs pillow numpy
  python train-model.py --pack morel --data ./training_data/morel

Training data structure:
  training_data/
    morel/
      morel_yellow/       <- 200+ images per class
        img001.jpg
        img002.jpg
      morel_grey/
      morel_black/
      false_morel_gyromitra/
      elm_dead/
      elm_living/
      ash_eab/
      ...
      background/

Image sources (public domain / licensed):
  - iNaturalist.org (research-grade observations)
  - Mushroom Observer (mushroomobserver.org)
  - USDA PLANTS database
  - Your own field photos from MorelEye users (with consent)
"""

import argparse
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflowjs as tfjs
from pathlib import Path

# ── Configuration ──────────────────────────────────
BACKBONE_URL = "https://tfhub.dev/google/imagenet/mobilenet_v3_small_100_224/feature_vector/5"
IMAGE_SIZE   = (224, 224)
BATCH_SIZE   = 32
EPOCHS       = 30
LEARNING_RATE = 0.001

# ── Species pack definitions ───────────────────────
PACK_CLASSES = {
    'morel': [
        'morel_yellow', 'morel_grey', 'morel_black', 'morel_half_free',
        'false_morel_gyromitra', 'elm_dead', 'elm_living', 'ash_eab',
        'apple_tree', 'tulip_poplar', 'sycamore', 'oak_mature',
        'favorable_habitat', 'leaf_litter', 'background'
    ],
    'chicken': [
        'chicken_woods_fresh', 'chicken_woods_old', 'jack_o_lantern',
        'oak_host', 'cherry_host', 'locust_host', 'background'
    ],
    'chanterelle': [
        'chanterelle_golden', 'chanterelle_cinnabar', 'jack_o_lantern_danger',
        'false_chanterelle', 'oak_beech_forest', 'background'
    ],
    'oyster': [
        'oyster_grey', 'oyster_golden', 'oyster_pink',
        'angel_wings_toxic', 'dead_hardwood', 'background'
    ],
    'ramps': [
        'ramps_leaves', 'ramps_bulb', 'lily_of_valley_toxic',
        'wild_onion', 'background'
    ],
    'berries': [
        'blackberry', 'blueberry', 'elderberry', 'serviceberry',
        'raspberry', 'pokeweed_toxic', 'nightshade_toxic', 'background'
    ],
}

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Build transfer learning model:
    MobileNet v3 Small (frozen) + custom classification head
    """
    # Load pretrained backbone from TF Hub
    backbone = hub.KerasLayer(
        BACKBONE_URL,
        input_shape=(*IMAGE_SIZE, 3),
        trainable=False,     # Freeze backbone initially
        name='mobilenet_v3'
    )

    # Classification head
    model = tf.keras.Sequential([
        backbone,
        tf.keras.layers.Dense(256, activation='relu', name='dense1'),
        tf.keras.layers.BatchNormalization(name='bn1'),
        tf.keras.layers.Dropout(0.4, name='dropout1'),
        tf.keras.layers.Dense(128, activation='relu', name='dense2'),
        tf.keras.layers.Dropout(0.3, name='dropout2'),
        tf.keras.layers.Dense(num_classes, activation='softmax', name='output')
    ], name='moreleye_classifier')

    return model


def load_dataset(data_dir: str, classes: list):
    """
    Load image dataset with augmentation for training.
    Returns (train_ds, val_ds)
    """
    # Data augmentation for robustness in field conditions
    # (varied lighting, angles, partial occlusion)
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomBrightness(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ], name='augmentation')

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=classes,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        class_names=classes,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    # Normalize to [0, 1] and apply augmentation to training only
    train_ds = train_ds.map(
        lambda x, y: (augmentation(x / 255.0, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    val_ds = val_ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds


def train_pack(pack_id: str, data_dir: str, output_dir: str):
    """Train and export a single species pack model."""

    classes = PACK_CLASSES.get(pack_id)
    if not classes:
        raise ValueError(f"Unknown pack: {pack_id}")

    print(f"\n{'='*50}")
    print(f"Training pack: {pack_id}")
    print(f"Classes: {len(classes)}")
    print(f"Data dir: {data_dir}")
    print(f"{'='*50}\n")

    # Load data
    train_ds, val_ds = load_dataset(data_dir, classes)

    # Build model
    model = build_model(len(classes))
    model.summary()

    # Phase 1: Train head only (backbone frozen)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, f'{pack_id}_best.h5'),
            monitor='val_accuracy', save_best_only=True
        ),
    ]

    print("Phase 1: Training classification head...")
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks
    )

    # Phase 2: Fine-tune top layers of backbone
    print("\nPhase 2: Fine-tuning top backbone layers...")
    # Unfreeze the backbone
    model.layers[0].trainable = True
    # Only train last 20 layers of MobileNet
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=10, callbacks=callbacks
    )

    # Evaluate
    print("\nFinal evaluation:")
    val_loss, val_acc, val_top2 = model.evaluate(val_ds)
    print(f"Validation accuracy: {val_acc:.3f}")
    print(f"Top-2 accuracy:      {val_top2:.3f}")

    if val_acc < 0.75:
        print(f"WARNING: Accuracy {val_acc:.1%} is below 75% threshold.")
        print("Consider adding more training images or adjusting hyperparameters.")

    # Export to TF.js format for on-device inference
    output_path = os.path.join(output_dir, f'{pack_id}-tfjs')
    os.makedirs(output_path, exist_ok=True)

    tfjs.converters.save_keras_model(
        model, output_path,
        quantization_dtype=np.uint8,  # 8-bit quantization — 4x smaller
        weight_shard_size_bytes=4 * 1024 * 1024  # 4MB shards
    )

    # Save class labels alongside model
    labels_path = os.path.join(output_path, 'labels.json')
    import json
    with open(labels_path, 'w') as f:
        json.dump({'pack': pack_id, 'classes': classes}, f, indent=2)

    print(f"\nModel exported to: {output_path}")
    print(f"Quantized model size: ~{estimate_size(output_path):.1f} MB")

    return val_acc


def estimate_size(directory: str) -> float:
    """Estimate total model size in MB."""
    total = sum(
        os.path.getsize(os.path.join(dirpath, f))
        for dirpath, _, files in os.walk(directory)
        for f in files
    )
    return total / (1024 * 1024)


def main():
    parser = argparse.ArgumentParser(description='Train MorelEye species pack models')
    parser.add_argument('--pack', choices=list(PACK_CLASSES.keys()) + ['all'],
                        default='morel', help='Which pack to train')
    parser.add_argument('--data', required=True,
                        help='Path to training data directory')
    parser.add_argument('--output', default='./trained_models',
                        help='Output directory for TF.js models')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    packs_to_train = list(PACK_CLASSES.keys()) if args.pack == 'all' else [args.pack]

    results = {}
    for pack_id in packs_to_train:
        pack_data_dir = os.path.join(args.data, pack_id)
        if not os.path.exists(pack_data_dir):
            print(f"WARNING: No data found for pack '{pack_id}' at {pack_data_dir}")
            continue
        acc = train_pack(pack_id, pack_data_dir, args.output)
        results[pack_id] = acc

    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    for pack, acc in results.items():
        status = "✓ PASS" if acc >= 0.75 else "✗ NEEDS MORE DATA"
        print(f"  {pack:15s}: {acc:.1%}  {status}")

    print(f"\nModels saved to: {args.output}")
    print("\nNext step: copy TF.js model folders to your app's /models/ directory")
    print("and update offline-engine.js modelFile paths accordingly.")


if __name__ == '__main__':
    main()
