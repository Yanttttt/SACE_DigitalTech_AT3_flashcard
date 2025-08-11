import { Draw, PhysicsScene, Entity, Collision, Joint, Vector2, VectorMath2 } from "./sp2d/sp2d.js";

var div = 2;

/**
 * 
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

    var ball = new Entity.Circle(0.1, 1, 0, 1,
        pos.add(new Vector2(-1.5/2, -1.5 * Math.sqrt(3)/2)),
        VectorMath2.zero(),
        0,
        0,
        "#7f3f00"
    );
    PhysicsScene.addEntity(ball);

    var rope = new Joint.Distance(hinge,
        ball,
        VectorMath2.zero(),
        VectorMath2.zero(),
        true,
        "#111111"
    );
    PhysicsScene.addJoint(rope);
}

function setupScene() {
    Draw.init("myCanvas", Math.min(window.innerWidth - 250, window.innerHeight - 250), Math.min(window.innerWidth - 250, window.innerHeight - 250), div);
    PhysicsScene.init(undefined, new Vector2(0.0, -9.8), 0.1);
    PhysicsScene.setFloorCollision(0, 0);

    var pos = new Vector2(0.5, 0.1);

    addPendulum(pos.add(new Vector2(0, 1.52)));

    var block1 = new Entity.Rectangle(
        0.2,
        0.2,
        1,//fully elastic
        0,
        1.2,//1 kg
        pos.add(new Vector2(0.2, 0)),
        new Vector2(0, 0),
        0,
        0,
        "#003f7f"
    );
    PhysicsScene.addEntity(block1);

    // pos = new Vector2(0.9, 0.1);
    // for (var i = 0; i < 3; i++) {
    //     var b = new Entity.Rectangle(
    //         0.2,
    //         0.2,
    //         1,
    //         0,
    //         1,//1 kg
    //         pos.add(new Vector2(i * 0.2, 0)),
    //         new Vector2(0, 0),
    //         0,
    //         0,
    //         Draw.getRandomColour()
    //     );
    //     PhysicsScene.addEntity(b);
    // }

    var block2 = new Entity.Rectangle(
        0.2,
        0.2,
        0,//fully elastic
        0,
        1.2,
        pos.add(new Vector2(0.8,0)),
        new Vector2(0, 0),
        0,
        0,
        "#5a7614ff"
    );
    PhysicsScene.addEntity(block2);
}
window.setupScene = setupScene;

function updateFrame() {
    PhysicsScene.simulate(100);
    PhysicsScene.draw();
    //equilibrium = new Vector2(0, 1.5).add(new Vector2(0, 0.196));

    requestAnimationFrame(updateFrame);
}

setupScene();
updateFrame();
