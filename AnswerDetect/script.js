const params = new URLSearchParams(window.location.search);
const page = params.get("page");
const question = params.get("question");
const sigfig = params.get("sigfig");

const path = "./model_4/weights.json";
var weights = null;
var canvases = [];
var ans = 0;

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

function createCanvas() {
    const container = document.createElement("div");
    container.style.margin = "10px";
    container.style.textAlign = "center";
    container.style.display = "inline-block";

    const canvas = document.createElement("canvas");
    canvas.width = 280;
    canvas.height = 280;
    canvas.style.border = "1px solid black";
    canvas.style.display = "block";
    canvas.style.marginBottom = "10px";

    const ctx = canvas.getContext("2d");
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

    const btn = document.createElement("button");
    btn.textContent = "Clear";

    const resultDiv = document.createElement("div");
    resultDiv.innerHTML = "Result: - <br>Also might be: -";

    btn.onclick = () => {
        ctx.fillStyle = 'white';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        resultDiv.innerHTML = "Result: -";
    };

    container.appendChild(canvas);
    container.appendChild(btn);
    container.appendChild(document.createElement("br"));
    container.appendChild(resultDiv);

    document.body.appendChild(container);

    return { "container": container, "canvas": canvas, "ctx": ctx, "resultDiv": resultDiv };
}

function getInput(canvas, ctx) {
    const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imgData.data;
    const input = new Float32Array(28 * 28);
    const scale = canvas.width / 28;

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

function predict(id) {
    let input = getInput(canvases[id].canvas, canvases[id].ctx);

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
        } else if (output.values[i] > output.values[secIdx]) {
            secIdx = i;
        }
    }

    if (output.values[maxIdx] > 0.15) {
        canvases[id].resultDiv.innerHTML =
            `Result: ${maxIdx} (prob: ${output.values[maxIdx].toFixed(4)})<br>
             Also might be: ${secIdx} (prob: ${output.values[secIdx].toFixed(4)})`;
    } else {
        canvases[id].resultDiv.innerHTML = "Result: - <br>Also might be: -";
        maxIdx = -1;
    }

    return maxIdx;
}

async function init() {
    weights = await loadWeights(path);
    console.log(weights);

    document.write("<h1>" + question + "</h1>");

    const totalDiv = document.createElement('div');
    totalDiv.id = 'ans';
    totalDiv.style.fontSize = '28px';
    totalDiv.style.marginTop = '16px';
    totalDiv.textContent = 'Your Answer is: (' + sigfig + " s.f.)";
    document.body.appendChild(totalDiv);

    const btn = document.createElement("button");
    btn.textContent = "Submit";
    btn.onclick = () => {
        window.location.href = page + "?ans=" + encodeURIComponent(ans)
    };
    document.body.appendChild(btn);
    document.body.appendChild(document.createElement("br"));

    for (let i = 0; i < sigfig; i++) {
        canvases.push(createCanvas());

        if (i === 0) {
            const dot = document.createElement("span");
            dot.style.display = "inline-block";
            dot.innerHTML = ".<br><br>";
            dot.style.fontSize = "70px";
            dot.style.margin = "0 10px";
            document.body.appendChild(dot);
        }
    }

    const times10 = document.createElement("span");
    times10.textContent = "x10";
    times10.style.fontSize = "124px";
    times10.style.margin = "0 10px";
    document.body.appendChild(times10);
    canvases.push(createCanvas());
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function main() {
    await init();
    while (1) {
        await sleep(500);
        let tot = "";
        ans = 0;
        for (let i = 0; i < canvases.length; i++) {
            let res = predict(i);
            if (i === 1) {
                tot += ".";
            }
            if (i === canvases.length - 1) {
                tot += "x10^" + res;
                ans*= Math.pow(10, res);
            }
            else if (res >= 0) {
                tot += res;
                ans += res / Math.pow(10, i);
            } else {
                tot += "?";
            }
        }
        console.log(ans);
        document.getElementById('ans').textContent = 'Your Answer is: ' + tot + " (" + sigfig + " s.f.)";
    }
}

main();
