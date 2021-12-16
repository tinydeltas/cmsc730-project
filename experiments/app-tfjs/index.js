import {L1} from './custom_layer.js';

let recognizer;

async function app() {
    recognizer = speechCommands.create('BROWSER_FFT');
    await recognizer.ensureModelLoaded();
    // buildModel();
    buildSiameseModel();
}

// One frame is ~23ms of audio.
const NUM_FRAMES = 3;
let examples = [];

function collect(label) {
    if (recognizer.isListening()) {
        return recognizer.stopListening();
    }
    if (label == null) {
        return;
    }
    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
        let vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        examples.push({vals, label});
        document.querySelector('#console').textContent = `${examples.length} examples collected`;
    }, {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
}

const INPUT_SHAPE = [NUM_FRAMES, 232, 1];
let model;

async function train() {
    toggleButtons(false);
    const ys = tf.oneHot(examples.map(e => e.label), 3);
    const xsShape = [examples.length, ...INPUT_SHAPE];
    const xs = tf.tensor(flatten(examples.map(e => e.vals)), xsShape);
    
    await model.fit(xs, ys, {
        batchSize: 16,
        epochs: 10,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                document.querySelector('#console').textContent =
                `Accuracy: ${(logs.acc * 100).toFixed(1)}% Epoch: ${epoch + 1}`;
            }
        }
    });
    tf.dispose([xs, ys]);
    toggleButtons(true);
}

function buildSiameseModel() {
    const inputShape = [ 100, 100, 3 ];
    const initializeWeights = tf.initializers.randomNormal({
        mean: 0.0, 
        stddev: 0.01, 
    });

    const initializeBias = tf.initializers.randomNormal({
        mean: 0.5, 
        stddev: 0.01, 
    });

    const regularizer = tf.regularizers.l2({
        l2: 2e-4
    });

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        filters: 64, 
        kernelSize: (10, 10), 
        activation: 'relu', 
        inputShape: inputShape, 
        kernelInitializer: initializeWeights, 
        kernelRegularizer: regularizer
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: (2, 2)
    }));

    model.add(tf.layers.conv2d({
        filters: 128, 
        kernelSize: (7, 7), 
        activation: 'relu', 
        kernelInitializer: initializeWeights, 
        biasInitializer: initializeBias, 
        kernelRegularizer: regularizer
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: (2, 2)
    }));

    model.add(tf.layers.conv2d({
        filters: 128, 
        kernelSize: (4, 4), 
        activation: 'relu', 
        kernelInitializer: initializeWeights, 
        biasInitializer: initializeBias, 
        kernelRegularizer: regularizer
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: (2, 2)
    }));

    model.add(tf.layers.conv2d({
        filters: 256,
        kernelSize: (4, 4), 
        activation: 'relu', 
        kernelInitializer: initializeWeights, 
        biasInitializer: initializeBias, 
        kernelRegularizer: regularizer
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 4096, 
        activation: 'sigmoid', 
        kernelRegularizer: tf.regularizers.l2({l2: 1e-3}),
        kernelInitializer: initializeWeights,
        biasInitializer: initializeBias, 
    }));

    const left = tf.input({
        shape:  [ 100, 100, 3 ]
    })

    const right = tf.input({
        shape:  [ 100, 100, 3 ]
    })

    // console.log("Test")
    // const encoded_l = tf.layers.input({
    //     inputs: left
    // }) 

    // const encoded_r = tf.layers.inputLayer({
    //     inputs: right 
    // })
    // console.log("Success")

    console.log("Left", left)
    console.log("Right", right)

    const l1_layer = new L1().apply({
        left: left, 
        right: right
    })

    const denseLayer = tf.layers.dense({
        units: 1,
        activation: 'sigmoid',
        biasInitializer: initializeBias
    });

    const prediction = denseLayer.apply(l1_layer); 

    model = tf.model({
        inputs: [left, right], 
        outputs: prediction
    });
    
    const optimizer = tf.train.adam(0.00006);
    
    model.compile({
        optimizer, 
        loss: 'binaryCrossentropy',
        metrics: ['accuracy'],
    });
}

function buildModel() {
    model = tf.sequential();
    model.add(tf.layers.depthwiseConv2d({
        depthMultiplier: 8,
        kernelSize: [NUM_FRAMES, 3],
        activation: 'relu',
        inputShape: INPUT_SHAPE
    }));
    model.add(tf.layers.maxPooling2d({poolSize: [1, 2], strides: [2, 2]}));
    model.add(tf.layers.flatten());
    model.add(tf.layers.dense({units: 3, activation: 'softmax'}));
    
    const optimizer = tf.train.adam(0.01);
    model.compile({
        optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });
}

function toggleButtons(enable) {
    document.querySelectorAll('button').forEach(b => b.disabled = !enable);
}

function flatten(tensors) {
    const size = tensors[0].length;
    const result = new Float32Array(tensors.length * size);
    tensors.forEach((arr, i) => result.set(arr, i * size));
    return result;
}

function normalize(x) {
    const mean = -100;
    const std = 10;
    return x.map(x => (x - mean) / std);
}

async function moveSlider(labelTensor) {
    const label = (await labelTensor.data())[0];
    document.getElementById('console').textContent = label;
    if (label == 2) {
        return;
    }
    let delta = 0.1;
    const prevValue = +document.getElementById('output').value;
    document.getElementById('output').value =
        prevValue + (label === 0 ? -delta : delta);
}

function listen() {
    if (recognizer.isListening()) {
        recognizer.stopListening();
        toggleButtons(true);
        document.getElementById('listen').textContent = 'Listen';
        return;
    }
    toggleButtons(false);
    document.getElementById('listen').textContent = 'Stop';
    document.getElementById('listen').disabled = false;


    recognizer.listen(async ({spectrogram: {frameSize, data}}) => {
        const vals = normalize(data.subarray(-frameSize * NUM_FRAMES));
        const input = tf.tensor(vals, [1, ...INPUT_SHAPE]);
        const probs = model.predict(input);
        const predLabel = probs.argMax(1);
        await moveSlider(predLabel);
        tf.dispose([input, probs, predLabel]);
    }, {
        overlapFactor: 0.999,
        includeSpectrogram: true,
        invokeCallbackOnNoiseAndUnknown: true
    });
}

app();

