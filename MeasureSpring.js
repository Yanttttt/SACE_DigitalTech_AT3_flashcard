import { Vector2, VectorMath2 } from "./sp2d/Vector2.js";
import * as Draw from "./sp2d/Draw.js";
import * as PhysicsScene from "./sp2d/PhysicsScene.js";
import * as Entity from "./sp2d/Entity.js";
import * as Collision from "./sp2d/Collision.js";
import * as Joint from "./sp2d/Joint.js";

var div = 2;
var timer = 0;
var paused = false;
var equilibrium = VectorMath2.zero();

function setupScene() {
    Draw.init("myCanvas", Math.min(window.innerWidth - 250, window.innerHeight - 250), Math.min(window.innerWidth - 250, window.innerHeight - 250), 2);
    PhysicsScene.init(undefined, new Vector2(0.0, -9.8), 0.1);
}
window.setupScene = setupScene;

/**
 * @param {Vector2} pos
 */
function addPendulum(pos) {
    var hinge = new Entity.Circle(0.01, 0, 0, Infinity,
        pos,
        VectorMath2.zero(),
        0,
        0,
        "#3f3f3f"
    )
    hinge.setStatic();
    PhysicsScene.addEntity(hinge);

    var length = 0.5;
    equilibrium = pos.add(new Vector2(0, -length - 0.196));

    var block = new Entity.Rectangle(
        0.1,
        0.1,
        1,
        0,
        1,//1 kg
        pos.add(new Vector2(0, -length - 0.1)),
        new Vector2(0, 0),
        0,
        0,
        "#7f3f00"
    );
    PhysicsScene.addEntity(block);

    var spring = new Joint.Spring(hinge, block,
        VectorMath2.zero(),
        new Vector2(0, 0.05),
        0.03,
        50,//50 N/m
        0,
        true,
        "#111111"
    );
    PhysicsScene.addJoint(spring);
}

function hangPendulum() {
    var mid = new Vector2(1, 1.5);
    addPendulum(mid);
}

function updateFrame() {
    if (paused) {
        requestAnimationFrame(updateFrame);
        return;
    }
    //console.log(PhysicsScene.entities);
    PhysicsScene.simulate(50);
    PhysicsScene.draw();
    //equilibrium = new Vector2(0, 1.5).add(new Vector2(0, 0.196));
    Draw.drawLine(
        equilibrium.add(new Vector2(-1, 0)),
        equilibrium.add(new Vector2(1, 0)),
        "#FF0000",
        1
    );

    requestAnimationFrame(updateFrame);
    timer += 1 / 60;
    document.getElementById("timer").textContent = timer.toFixed(2) + " s";
}

function togglePause() {
    console.log("togglePause", paused);
    if (paused) {
        document.getElementById("pauseButton").textContent = "||";
    }
    else {
        document.getElementById("pauseButton").textContent = "▶";
        var p = document.createElement("p");
        p.textContent = "Paused at " + timer.toFixed(2) + " s";
        document.getElementById("record").appendChild(p);
    }
    paused = !paused;
}
window.togglePause = togglePause;

function answer() {
    const question = "What is the spring constant of the spring in N/m? (2 significant fiures)";
    const sigfig = 2;

    window.location.href = "./AnswerDetect/index.html?page="+encodeURIComponent("../MeasureSpring.html") +"&question=" + encodeURIComponent(question) + "&sigfig=" + encodeURIComponent(sigfig);
}
window.answer = answer;

function checkAnswer() {
    const ans = (new URLSearchParams(window.location.search)).get("ans");
    if (!ans) return;
    const intans = parseInt(ans);
    const delta = 1;
    if (intans >= 50 - delta && intans <= 50 + delta) {
        alert("Correct! The spring constant is 50 N/m.");
    }
    else {
        alert("Incorrect. The spring constant is 50 N/m.");
    }
}

checkAnswer();
setupScene();
hangPendulum();
updateFrame();