 import { LayersModel, loadLayersModel, Rank, Tensor, tensor } from "@tensorflow/tfjs-node";
 import path from "path";
 import workerpool from "workerpool";

// variable to store the neural net
 let model: LayersModel;

// main function to be executed by the worker
 async function predict(pixels: number[]): Promise<string> {
    try {
        if (!model) {
            model = await loadLayersModel(
                "file://" + path.join(__dirname, "../", "../", "saved", "default", "model.json"));
        }

        const pixelTensor = tensor(pixels, [1, 28, 28, 1]);
        const resultTensor = model.predict(pixelTensor) as Tensor<Rank>;
        const resultVector = await resultTensor.argMax(1).array() as number[];
        return resultVector[0] + "";
    } catch (e) {
        return "undefined";
    }
}

 workerpool.worker({
    predict
});
