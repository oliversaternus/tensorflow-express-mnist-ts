"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const tfjs_node_1 = require("@tensorflow/tfjs-node");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
let model; // global variable to store the neural net
// defines the architecture of the neural net
function defineModel() {
    model = tfjs_node_1.sequential();
    model.add(tfjs_node_1.layers.conv2d({
        activation: "relu",
        filters: 8,
        inputShape: [28, 28, 1],
        kernelInitializer: "varianceScaling",
        kernelSize: 7,
        strides: 1
    }));
    model.add(tfjs_node_1.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tfjs_node_1.layers.conv2d({
        activation: "relu",
        filters: 12,
        kernelInitializer: "varianceScaling",
        kernelSize: 5,
        strides: 1,
    }));
    model.add(tfjs_node_1.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tfjs_node_1.layers.conv2d({
        activation: "relu",
        filters: 16,
        kernelInitializer: "varianceScaling",
        kernelSize: 3,
        strides: 1,
    }));
    model.add(tfjs_node_1.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));
    model.add(tfjs_node_1.layers.flatten());
    model.add(tfjs_node_1.layers.dense({ units: 3, activation: "softmax" }));
    const optimizer = tfjs_node_1.train.adam();
    model.compile({ optimizer, loss: "categoricalCrossentropy", metrics: ["accuracy"] });
}
// loads the MNIST data from harddrive and reshapes it into tensors
function openMNIST() {
    const tkI = fs_1.default.readFileSync(path_1.default.join(__dirname, "../", "/MNIST", "t10k-images.idx3-ubyte"));
    const tkL = fs_1.default.readFileSync(path_1.default.join(__dirname, "../", "/MNIST", "t10k-labels.idx1-ubyte"));
    const trainI = fs_1.default.readFileSync(path_1.default.join(__dirname, "../", "/MNIST", "train-images.idx3-ubyte"));
    const trainL = fs_1.default.readFileSync(path_1.default.join(__dirname, "../", "/MNIST", "train-labels.idx1-ubyte"));
    const testLabels = [...tkL].slice(8);
    const testImages = [...tkI].slice(16);
    const trainLabels = [...trainL].slice(8);
    const trainImages = [...trainI].slice(16);
    const trainInput = tfjs_node_1.tensor2d(trainImages, [60000, 784]).reshape([60000, 28, 28, 1]);
    const trainOutput = tfjs_node_1.oneHot(trainLabels, 10);
    const testInput = tfjs_node_1.tensor2d(testImages, [10000, 784]).reshape([10000, 28, 28, 1]);
    const testOutput = tfjs_node_1.oneHot(testLabels, 10);
    return { train: { input: trainInput, output: trainOutput }, test: { input: testInput, output: testOutput } };
}
// starts the training process
function trainModel(data, epochs) {
    return __awaiter(this, void 0, void 0, function* () {
        yield model.fit(data.input, data.output, {
            callbacks: {
                onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
            },
            epochs
        });
        try {
            yield model.save("file://" + path_1.default.join(__dirname, "../", "/saved", Date.now() + ""));
            console.log("model saved!");
        }
        catch (e) {
            console.log(e);
        }
    });
}
// tests the model using the test-dataset
function testModel(testData) {
    return __awaiter(this, void 0, void 0, function* () {
        const result = yield model.predict(testData.input).argMax(1).array();
        const output = yield testData.output.argMax(1).array();
        let errorCount = 0;
        for (let i = 0; i < output.length; i++) {
            if (output[i] !== result[i]) {
                errorCount += 1;
            }
        }
        console.log(errorCount + " errors, " + (errorCount * 100 / output.length) + "% error rate");
    });
}
// main function to run the script asynchronously
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const data = openMNIST();
        defineModel();
        yield trainModel(data.train, 100);
        yield testModel(data.test);
    });
}
// start
main();
//# sourceMappingURL=training.js.map