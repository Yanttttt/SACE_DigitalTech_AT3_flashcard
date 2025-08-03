// 加载模型权重
async function loadModel() {
  const res = await fetch('model.json');
  return await res.json(); // 包含 w1, b1, w2, b2
}

// sigmoid 激活函数
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

// softmax 输出层
function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b);
  return exps.map(e => e / sum);
}

// 前向传播：输入 → 隐藏层 → 输出层
function predict(input, model) {
  const hidden = [];
  for (let i = 0; i < model.b1.length; i++) {
    let sum = model.b1[i];
    for (let j = 0; j < input.length; j++) {
      sum += input[j] * model.w1[i][j];
    }
    hidden[i] = sigmoid(sum);
  }

  const output = [];
  for (let i = 0; i < model.b2.length; i++) {
    let sum = model.b2[i];
    for (let j = 0; j < hidden.length; j++) {
      sum += hidden[j] * model.w2[i][j];
    }
    output[i] = sum;
  }

  return softmax(output);
}

// 将 canvas 缩小为 28x28 并转换为灰度数组
function getInputFromCanvas(canvas) {
  const temp = document.createElement('canvas');
  temp.width = 28;
  temp.height = 28;
  const tempCtx = temp.getContext('2d');
  tempCtx.drawImage(canvas, 0, 0, 28, 28);

  const imgData = tempCtx.getImageData(0, 0, 28, 28).data;
  const input = [];
  for (let i = 0; i < imgData.length; i += 4) {
    const grayscale = 255 - imgData[i]; // 反色：白底黑字 → 黑底白字
    input.push(grayscale / 255);
  }
  return input;
}
