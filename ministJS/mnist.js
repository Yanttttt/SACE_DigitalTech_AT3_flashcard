const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

let painting = false;
canvas.addEventListener('mousedown', () => painting = true);
canvas.addEventListener('mouseup', () => { painting = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(e) {
  if (!painting) return;
  const rect = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
}

async function runPrediction() {
  const input = getInputFromCanvas(canvas); // 获取28x28灰度输入
  const model = await loadModel();          // 加载模型
  const output = predict(input, model);     // 前向传播
  const pred = output.indexOf(Math.max(...output));
  document.getElementById("result").textContent = `预测结果：${pred}`;
}
