var path = "./model_4/weights.json";
var canvas = null;
var ctx = null;
var weights = null;

async function loadWeights(path) {
    const res = await fetch(path);
    const weightsJson = await res.json();
    const weightMap = {};
    for (const w of weightsJson) {
        weightMap[w.name] = {
            shape: w.shape,
            values: new Float32Array(w.data)
        };
    }
    return weightMap;
}

function setupCanvas() {
    canvas = /** @type {HTMLCanvasElement} */ (document.getElementById('canvas'));
    ctx = /** @type {HTMLCanvasElement} */ (canvas).getContext("2d");

    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    let drawing = false;
    canvas.onmousedown = () => drawing = true;
    canvas.onmouseup = () => drawing = false;
    canvas.onmouseleave = () => drawing = false;
    canvas.onmousemove = e => {
        if (!drawing) return;
        ctx.beginPath();
        ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
        ctx.fillStyle = 'black';
        ctx.fill();
    };
}

function clear() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}
window.clear = clear;


/**
 * @param {{ width: number; height: any; }} canvas
 */
function getInput(canvas) {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const input = new Float32Array(28 * 28);
    const scale = canvas.width / 28; // 10 if 280x280

    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let sum = 0;
            for (let dy = 0; dy < scale; dy++) {
                for (let dx = 0; dx < scale; dx++) {
                    const px = Math.floor(x * scale + dx);
                    const py = Math.floor(y * scale + dy);
                    const idx = (py * canvas.width + px) * 4;
                    sum += data[idx];
                }
            }
            const avg = sum / (scale * scale);
            input[y * 28 + x] = (255 - avg) / 255;
        }
    }



    // const outCanvas = document.createElement('canvas');
    // outCanvas.width = 28;
    // outCanvas.height = 28;
    // const outCtx = outCanvas.getContext('2d');

    // const imageData = outCtx.createImageData(28, 28);
    // for (let i = 0; i < input.length; i++) {
    //     const color = input[i] * 255; // 0 ~ 255
    //     imageData.data[i * 4 + 0] = color; // R
    //     imageData.data[i * 4 + 1] = color; // G
    //     imageData.data[i * 4 + 2] = color; // B
    //     imageData.data[i * 4 + 3] = 255;   // Alpha
    // }

    // outCtx.putImageData(imageData, 0, 0);

    // const bigCanvas = document.createElement('canvas');
    // bigCanvas.width = 280;
    // bigCanvas.height = 280;
    // const bigCtx = bigCanvas.getContext('2d');
    // bigCtx.imageSmoothingEnabled = false;
    // bigCtx.drawImage(outCanvas, 0, 0, 280, 280);
    // document.body.appendChild(bigCanvas);

    return { shape: [28, 28, 1], values: input };
}

function conv2d(input, kernel, bias) {
    const [kh, kw, cin, cout] = kernel.shape;
    const [h, w, channels] = input.shape;
    const oh = h - kh + 1;
    const ow = w - kw + 1;
    const output = new Float32Array(oh * ow * cout);

    for (let co = 0; co < cout; co++) {
        for (let y = 0; y < oh; y++) {
            for (let x = 0; x < ow; x++) {
                let sum = bias.values[co];
                for (let ky = 0; ky < kh; ky++) {
                    for (let kx = 0; kx < kw; kx++) {
                        for (let ci = 0; ci < cin; ci++) {
                            const inIdx = ((y + ky) * w + (x + kx)) * channels + ci;
                            const kIdx = ((ky * kw + kx) * cin + ci) * cout + co;
                            sum += input.values[inIdx] * kernel.values[kIdx];
                        }
                    }
                }
                const outIdx = (y * ow + x) * cout + co;
                output[outIdx] = Math.max(0, sum);
            }
        }
    }
    return { shape: [oh, ow, cout], values: output };
}

function maxPool(input) {
    const [h, w, c] = input.shape;
    const oh = Math.floor(h / 2);
    const ow = Math.floor(w / 2);
    const output = new Float32Array(oh * ow * c);

    for (let ci = 0; ci < c; ci++) {
        for (let y = 0; y < oh; y++) {
            for (let x = 0; x < ow; x++) {
                let maxVal = -Infinity;
                for (let py = 0; py < 2; py++) {
                    for (let px = 0; px < 2; px++) {
                        const idx = ((y * 2 + py) * w + (x * 2 + px)) * c + ci;
                        maxVal = Math.max(maxVal, input.values[idx]);
                    }
                }
                output[(y * ow + x) * c + ci] = maxVal;
            }
        }
    }
    return { shape: [oh, ow, c], values: output };
}

function flatten(input) {
    return { shape: [input.values.length], values: input.values };
}

function dense(input, weights, bias) {
    const [inSize] = input.shape;
    const [wIn, wOut] = weights.shape;
    const output = new Float32Array(wOut);

    for (let o = 0; o < wOut; o++) {
        let sum = bias.values[o];
        for (let i = 0; i < inSize; i++) {
            sum += input.values[i] * weights.values[i * wOut + o];
        }
        output[o] = sum;
    }
    // softmax
    const maxVal = Math.max(...output);
    let sumExp = 0;
    for (let i = 0; i < output.length; i++) {
        output[i] = Math.exp(output[i] - maxVal);
        sumExp += output[i];
    }
    for (let i = 0; i < output.length; i++) {
        output[i] /= sumExp;
    }
    return { shape: [wOut], values: output };
}


function predict() {
    if (!weights) {
        alert("model not loaded");
    }

    let input = getInput(canvas);

    input = conv2d(input, weights["conv2d_Conv2D1/kernel"], weights["conv2d_Conv2D1/bias"]);
    input = maxPool(input);

    input = conv2d(input, weights["conv2d_Conv2D2/kernel"], weights["conv2d_Conv2D2/bias"]);
    input = maxPool(input);

    input = flatten(input);

    const output = dense(input, weights["dense_Dense1/kernel"], weights["dense_Dense1/bias"]);

    let maxIdx = 0;
    let secIdx = 0;
    for (let i = 1; i < output.values.length; i++) {
        if (output.values[i] > output.values[maxIdx]) {
            secIdx = maxIdx;
            maxIdx = i;
        }
        else if (output.values[i] > output.values[secIdx]) {
            secIdx = i;
        }
    }

    if (output.values[maxIdx] > 0.15) {
        document.getElementById('result').innerHTML =
            `Result: ${maxIdx} (prob: ${output.values[maxIdx].toFixed(4)})<br>
         Also might be: ${secIdx} (prob: ${output.values[secIdx].toFixed(4)})`;
    }

}
window.predict = predict;

async function init() {
    weights = await loadWeights(path);
    console.log(weights);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}
async function main() {
    await init();
    setupCanvas();
    clear();
    document.getElementById("clear").addEventListener("click", clear);

    while (1) {
        await sleep(500);
        predict();
    }
}

main();

