import { layers, oneHot, Rank, Sequential, sequential, Tensor, tensor2d, train } from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";

interface Data {
    input: Tensor<Rank>;
    output: Tensor<Rank>;
}

let model: Sequential; // global variable to store the neural net

// defines the architecture of the neural net
function defineModel() {
    model = sequential();
    model.add(layers.conv2d({
        activation: "relu",
        filters: 8,
        inputShape: [28, 28, 1],
        kernelInitializer: "varianceScaling",
        kernelSize: 7,
        strides: 1
    }));
    model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(layers.conv2d({
        activation: "relu",
        filters: 12,
        kernelInitializer: "varianceScaling",
        kernelSize: 5,
        strides: 1,
    }));
    model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(layers.conv2d({
        activation: "relu",
        filters: 16,
        kernelInitializer: "varianceScaling",
        kernelSize: 3,
        strides: 1,
    }));
    model.add(layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(layers.flatten());
    model.add(layers.dense({ units: 3, activation: "softmax" }));
    const optimizer = train.adam();
    model.compile({ optimizer, loss: "categoricalCrossentropy", metrics: ["accuracy"] });
}

// loads the MNIST data from harddrive and reshapes it into tensors
function openMNIST() {
    const tkI = fs.readFileSync(path.join(__dirname, "../", "/MNIST", "t10k-images.idx3-ubyte"));
    const tkL = fs.readFileSync(path.join(__dirname, "../", "/MNIST", "t10k-labels.idx1-ubyte"));
    const trainI = fs.readFileSync(path.join(__dirname, "../", "/MNIST", "train-images.idx3-ubyte"));
    const trainL = fs.readFileSync(path.join(__dirname, "../", "/MNIST", "train-labels.idx1-ubyte"));

    const testLabels = [...tkL].slice(8);
    const testImages = [...tkI].slice(16);
    const trainLabels = [...trainL].slice(8);
    const trainImages = [...trainI].slice(16);

    const trainInput = tensor2d(trainImages, [60000, 784]).reshape([60000, 28, 28, 1]);
    const trainOutput = oneHot(trainLabels, 10);
    const testInput = tensor2d(testImages, [10000, 784]).reshape([10000, 28, 28, 1]);
    const testOutput = oneHot(testLabels, 10);

    return { train: { input: trainInput, output: trainOutput }, test: { input: testInput, output: testOutput } };
}

// starts the training process
async function trainModel(data: Data, epochs: number) {
    await model.fit(data.input, data.output, {
        callbacks: {
            onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
        },
        epochs
    });
    try {
        await model.save("file://" + path.join(__dirname, "../", "/saved", Date.now() + ""));
        console.log("model saved!");
    } catch (e) {
        console.log(e);
    }
}

// tests the model using the test-dataset
async function testModel(testData: Data) {
    const result: number[] = await (model.predict(testData.input) as Tensor<Rank>).argMax(1).array() as number[];
    const output: number[] = await testData.output.argMax(1).array() as number[];
    let errorCount = 0;
    for (let i = 0; i < output.length; i++) {
        if (output[i] !== result[i]) {
            errorCount += 1;
        }
    }
    console.log(errorCount + " errors, " + (errorCount * 100 / output.length) + "% error rate");
}

// main function to run the script asynchronously
async function main() {
    const data = openMNIST();
    defineModel();
    await trainModel(data.train, 100);
    await testModel(data.test);
}

// start
main();
