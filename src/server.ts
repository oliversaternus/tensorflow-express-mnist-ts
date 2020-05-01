import bodyParser from "body-parser";
import cors from "cors";
import express from "express";
import { createServer } from "http";
import path from "path";
import workerpool from "workerpool";
import autoCatch from "./tools/catch";

const app: express.Application = express();
const pool = workerpool.pool(__dirname + "/tools/worker.js");

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(cors());

app.use(express.static(path.join(__dirname, "../", "assets")));

app.get("/", autoCatch(async (req, res) => {
    res.sendFile(path.join(__dirname, "../", "index.html"));
}));

app.post("/classify",
    autoCatch(async (req, res) => {
        const pixels = req.body as number[];
        const prediction: string = await pool.exec("predict", [pixels]);
        res.status(200).send({ prediction });
    }));

app.use((err: any, req: any, res: any, next: any) => {
    console.log(err);
    res.status(Number(err.message) || 500);
    res.send();
});

const server = createServer(app);
server.listen(5454, () => {
    console.log(`MNIST image classifying server started at http://localhost:5454`);
});
