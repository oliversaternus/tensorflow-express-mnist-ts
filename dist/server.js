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
const body_parser_1 = __importDefault(require("body-parser"));
const cors_1 = __importDefault(require("cors"));
const express_1 = __importDefault(require("express"));
const http_1 = require("http");
const path_1 = __importDefault(require("path"));
const workerpool_1 = __importDefault(require("workerpool"));
const catch_1 = __importDefault(require("./tools/catch"));
const app = express_1.default();
const pool = workerpool_1.default.pool(__dirname + "/worker.js");
app.use(body_parser_1.default.urlencoded({ extended: true }));
app.use(body_parser_1.default.json());
app.use(cors_1.default());
app.get("/", catch_1.default((req, res) => __awaiter(void 0, void 0, void 0, function* () {
    res.sendFile(path_1.default.join(__dirname, "../", "assets", "index.html"));
})));
app.get("/dist/bundle.js", catch_1.default((req, res) => __awaiter(void 0, void 0, void 0, function* () {
    res.sendFile(path_1.default.join(__dirname, "../", "assets", "bundle.js"));
})));
// #############################################################################
// classify image data
app.post("/classify", catch_1.default((req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const pixels = req.body;
    const prediction = yield pool.exec("predict", [pixels]);
    res.status(200).send({ prediction });
})));
// #############################################################################
app.use((err, req, res, next) => {
    console.log(err);
    res.status(Number(err.message) || 500);
    res.send();
});
// #############################################################################
const server = http_1.createServer(app);
server.listen(5454, () => {
    console.log(`MNIST image classifying server started at http://localhost:5454`);
});
//# sourceMappingURL=server.js.map