import cv2
import tensorflow as tf
import numpy as np
import glob

trainInput = []
trainOutput = []

testInput = []
testOutput = []


for img in glob.glob("iamges/*.jpg"):
    cv_img = cv2.imread(img, 1)
    resized_img = cv2.resize(cv_img, (255, 255))

    norm = [
        [[0 for k in range(len(resized_img[0][0]))] for j in range(len(resized_img[0]))]
        for i in range(len(resized_img))
    ]
    # aduc datele in intervalul [0,1]
    for i in range(len(resized_img)):
        for j in range(len(resized_img[0])):
            for k in range(len(resized_img[0][0])):
                norm[i][j][k] = resized_img[i][j][k] / 255

    if "train" in img:
        trainInput.append(resized_img)

        if "sepia" in img:
            trainOutput.append(1)
        else:
            trainOutput.append(0)
    else:
        testInput.append(resized_img)

        if "sepia" in img:
            testOutput.append(1)
        else:
            testOutput.append(0)

# creez un kereas.Sequential pentru prob de clasificare
modelTensor = tf.keras.Sequential(
    [
        # ca si atribute am functia de activare nr de neuroni nr de kernal si inputul o matrice 3D de 255 X 255 X 3
        tf.keras.layers.Conv2D(
            64, (3, 3), activation="sigmoid", input_shape=(255, 255, 3)
        ),
        tf.keras.layers.Conv2D(32, (3, 3), activation="sigmoid"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation="sigmoid"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# ii dau datele cu care trebuie sa compileze
modelTensor.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss="mean_absolute_error",
    metrics=["accuracy"],
)

inputs = np.asarray(trainInput)
outputs = np.asarray(trainOutput)
# se face un fit pe model
history = modelTensor.fit(inputs, outputs, batch_size=10, epochs=4)

print("History:", history.history)

# tr sa folosesc np.asarray deoarece modelTensor doar astefl de input accepta
testIn = np.asarray(testInput)
testOut = np.asarray(testOutput)

test_loss, test_acc = modelTensor.evaluate(testIn, testOut)
print(test_loss, test_acc)
